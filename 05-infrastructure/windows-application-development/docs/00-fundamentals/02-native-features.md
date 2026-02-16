# ネイティブ機能の活用

> デスクトップアプリの真価はネイティブ機能にある。ファイルシステムアクセス、システム通知、トレイアイコン、自動起動、グローバルショートカット、クリップボード、ドラッグ&ドロップまで、OS との深い統合を解説する。さらに .NET デスクトップ（WPF/WinUI 3）における Win32 API 呼び出し、レジストリ操作、Windows サービス連携、タスクスケジューラ登録、シェル統合まで包括的にカバーする。

## この章で学ぶこと

- [ ] ファイルダイアログとファイルシステム操作を実装できる
- [ ] システム通知・トレイアイコンを活用できる
- [ ] グローバルショートカット・自動起動を設定できる
- [ ] Win32 API を P/Invoke で呼び出す方法を理解する
- [ ] レジストリ操作・環境変数管理ができる
- [ ] シェル統合（コンテキストメニュー・ファイル関連付け）を実装できる
- [ ] Windows サービスとの連携パターンを理解する

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

### 1.1 Electron 実装

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

// 複数ファイルの一括選択
ipcMain.handle('file:openMultiple', async () => {
  const result = await dialog.showOpenDialog({
    title: '複数ファイルを開く',
    filters: [
      { name: '画像', extensions: ['png', 'jpg', 'jpeg', 'gif', 'webp'] },
    ],
    properties: ['openFile', 'multiSelections'],
  });

  if (result.canceled) return [];

  const files = await Promise.all(
    result.filePaths.map(async (filePath) => {
      const stat = await fs.stat(filePath);
      return {
        path: filePath,
        name: path.basename(filePath),
        size: stat.size,
        lastModified: stat.mtime,
      };
    })
  );

  return files;
});

// ディレクトリ選択
ipcMain.handle('file:selectDirectory', async () => {
  const result = await dialog.showOpenDialog({
    title: 'フォルダを選択',
    properties: ['openDirectory', 'createDirectory'],
  });

  if (result.canceled || !result.filePaths[0]) return null;
  return result.filePaths[0];
});
```

### 1.2 Tauri 実装

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

### 1.3 WPF/WinUI 3 でのファイルダイアログ

```csharp
// WPF — Microsoft.Win32 のファイルダイアログ
using Microsoft.Win32;

public class FileDialogService : IFileDialogService
{
    public string? OpenFile(string title, string filter)
    {
        var dialog = new OpenFileDialog
        {
            Title = title,
            Filter = filter,
            // 例: "テキストファイル (*.txt)|*.txt|すべてのファイル (*.*)|*.*"
            CheckFileExists = true,
            Multiselect = false,
            InitialDirectory = Environment.GetFolderPath(
                Environment.SpecialFolder.MyDocuments),
        };

        return dialog.ShowDialog() == true ? dialog.FileName : null;
    }

    public string[]? OpenFiles(string title, string filter)
    {
        var dialog = new OpenFileDialog
        {
            Title = title,
            Filter = filter,
            Multiselect = true,
        };

        return dialog.ShowDialog() == true ? dialog.FileNames : null;
    }

    public string? SaveFile(string title, string filter, string defaultFileName)
    {
        var dialog = new SaveFileDialog
        {
            Title = title,
            Filter = filter,
            FileName = defaultFileName,
            OverwritePrompt = true,
        };

        return dialog.ShowDialog() == true ? dialog.FileName : null;
    }

    public string? SelectFolder(string title)
    {
        // .NET 8 以降は FolderBrowserDialog が改良されている
        var dialog = new OpenFolderDialog
        {
            Title = title,
            Multiselect = false,
        };

        return dialog.ShowDialog() == true ? dialog.FolderName : null;
    }
}
```

```csharp
// WinUI 3 — Windows.Storage.Pickers
using Windows.Storage.Pickers;
using WinRT.Interop;

public class WinUIFileDialogService : IFileDialogService
{
    private readonly Window _window;

    public WinUIFileDialogService(Window window)
    {
        _window = window;
    }

    public async Task<StorageFile?> OpenFileAsync()
    {
        var picker = new FileOpenPicker();
        picker.FileTypeFilter.Add(".txt");
        picker.FileTypeFilter.Add(".md");
        picker.FileTypeFilter.Add(".json");

        // WinUI 3 では HWND の設定が必要
        var hwnd = WindowNative.GetWindowHandle(_window);
        InitializeWithWindow.Initialize(picker, hwnd);

        return await picker.PickSingleFileAsync();
    }

    public async Task<StorageFile?> SaveFileAsync(string suggestedName)
    {
        var picker = new FileSavePicker();
        picker.SuggestedFileName = suggestedName;
        picker.FileTypeChoices.Add("Markdown", new List<string> { ".md" });
        picker.FileTypeChoices.Add("テキスト", new List<string> { ".txt" });

        var hwnd = WindowNative.GetWindowHandle(_window);
        InitializeWithWindow.Initialize(picker, hwnd);

        return await picker.PickSaveFileAsync();
    }

    public async Task<StorageFolder?> SelectFolderAsync()
    {
        var picker = new FolderPicker();
        picker.FileTypeFilter.Add("*");

        var hwnd = WindowNative.GetWindowHandle(_window);
        InitializeWithWindow.Initialize(picker, hwnd);

        return await picker.PickSingleFolderAsync();
    }
}
```

### 1.4 ファイル監視（FileSystemWatcher）

```csharp
// ファイルシステムの変更をリアルタイム監視
using System.IO;

public class FileWatcherService : IDisposable
{
    private readonly FileSystemWatcher _watcher;
    private readonly Subject<FileChangeEvent> _changes = new();

    public IObservable<FileChangeEvent> Changes => _changes.AsObservable();

    public FileWatcherService(string directoryPath, string filter = "*.*")
    {
        _watcher = new FileSystemWatcher(directoryPath)
        {
            Filter = filter,
            NotifyFilter = NotifyFilters.FileName
                | NotifyFilters.LastWrite
                | NotifyFilters.Size
                | NotifyFilters.CreationTime,
            IncludeSubdirectories = true,
            EnableRaisingEvents = true,
        };

        _watcher.Created += (s, e) => _changes.OnNext(
            new FileChangeEvent(e.FullPath, FileChangeType.Created));
        _watcher.Changed += (s, e) => _changes.OnNext(
            new FileChangeEvent(e.FullPath, FileChangeType.Modified));
        _watcher.Deleted += (s, e) => _changes.OnNext(
            new FileChangeEvent(e.FullPath, FileChangeType.Deleted));
        _watcher.Renamed += (s, e) => _changes.OnNext(
            new FileChangeEvent(e.FullPath, FileChangeType.Renamed, e.OldFullPath));
        _watcher.Error += (s, e) => _changes.OnError(e.GetException());
    }

    public void Dispose()
    {
        _watcher.EnableRaisingEvents = false;
        _watcher.Dispose();
        _changes.Dispose();
    }
}

public record FileChangeEvent(
    string Path,
    FileChangeType Type,
    string? OldPath = null);

public enum FileChangeType { Created, Modified, Deleted, Renamed }
```

```csharp
// Electron — ファイル監視
// main.ts
import { watch } from 'chokidar';

const watcher = watch('/path/to/watch', {
  persistent: true,
  ignoreInitial: true,
  awaitWriteFinish: {
    stabilityThreshold: 300,
    pollInterval: 100,
  },
});

watcher.on('change', (path) => {
  mainWindow?.webContents.send('file:changed', { path, type: 'change' });
});
watcher.on('add', (path) => {
  mainWindow?.webContents.send('file:changed', { path, type: 'add' });
});
watcher.on('unlink', (path) => {
  mainWindow?.webContents.send('file:changed', { path, type: 'unlink' });
});
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

### 2.1 .NET デスクトップ通知

```csharp
// WinUI 3 — AppNotificationManager を使ったトースト通知
using Microsoft.Windows.AppNotifications;
using Microsoft.Windows.AppNotifications.Builder;

public class NotificationService
{
    public void Initialize()
    {
        // 通知マネージャーの初期化
        AppNotificationManager.Default.NotificationInvoked += OnNotificationInvoked;
        AppNotificationManager.Default.Register();
    }

    /// <summary>
    /// シンプルなトースト通知を表示
    /// </summary>
    public void ShowSimple(string title, string message)
    {
        var builder = new AppNotificationBuilder()
            .AddText(title)
            .AddText(message);

        var notification = builder.BuildNotification();
        AppNotificationManager.Default.Show(notification);
    }

    /// <summary>
    /// アクションボタン付きトースト通知
    /// </summary>
    public void ShowWithActions(string title, string message,
        params (string Label, string ActionId)[] actions)
    {
        var builder = new AppNotificationBuilder()
            .AddText(title)
            .AddText(message);

        foreach (var (label, actionId) in actions)
        {
            builder.AddButton(new AppNotificationButton(label)
                .AddArgument("action", actionId));
        }

        var notification = builder.BuildNotification();
        AppNotificationManager.Default.Show(notification);
    }

    /// <summary>
    /// 進捗バー付き通知
    /// </summary>
    public void ShowProgress(string title, double progress, string status)
    {
        var builder = new AppNotificationBuilder()
            .AddText(title)
            .AddProgressBar(new AppNotificationProgressBar()
            {
                Title = "ダウンロード中",
                Value = progress,
                ValueStringOverride = $"{progress * 100:F0}%",
                Status = status,
            });

        var notification = builder.BuildNotification();
        notification.Tag = "download-progress";
        notification.Group = "downloads";
        AppNotificationManager.Default.Show(notification);
    }

    /// <summary>
    /// 画像付き通知
    /// </summary>
    public void ShowWithImage(string title, string message, string imagePath)
    {
        var builder = new AppNotificationBuilder()
            .AddText(title)
            .AddText(message)
            .SetInlineImage(new Uri(imagePath));

        var notification = builder.BuildNotification();
        AppNotificationManager.Default.Show(notification);
    }

    private void OnNotificationInvoked(
        AppNotificationManager sender,
        AppNotificationActivatedEventArgs args)
    {
        // 通知クリック時またはアクションボタン押下時の処理
        var actionId = args.Arguments.ContainsKey("action")
            ? args.Arguments["action"]
            : "default";

        // UI スレッドで処理
        App.MainWindow.DispatcherQueue.TryEnqueue(() =>
        {
            HandleNotificationAction(actionId);
        });
    }

    private void HandleNotificationAction(string actionId)
    {
        switch (actionId)
        {
            case "open-file":
                // ファイルを開く処理
                break;
            case "dismiss":
                // 通知を閉じる
                break;
            default:
                // アプリをフォアグラウンドに
                App.MainWindow.Activate();
                break;
        }
    }

    public void Cleanup()
    {
        AppNotificationManager.Default.Unregister();
    }
}
```

```xml
<!-- XML ベースのトースト通知テンプレート（高度なカスタマイズ） -->
<!--
<toast launch="action=viewConversation&amp;conversationId=9813">
  <visual>
    <binding template="ToastGeneric">
      <text>新着メッセージ</text>
      <text>田中さんからメッセージが届きました</text>
      <image placement="appLogoOverride"
             hint-crop="circle"
             src="ms-appx:///Assets/user-avatar.png"/>
    </binding>
  </visual>
  <actions>
    <input id="replyBox" type="text" placeHolderContent="返信を入力..." />
    <action content="送信"
            arguments="action=reply&amp;conversationId=9813"
            activationType="background"
            hint-inputId="replyBox" />
    <action content="既読にする"
            arguments="action=markRead&amp;conversationId=9813"
            activationType="background" />
  </actions>
  <audio src="ms-winsoundevent:Notification.IM" />
</toast>
-->
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

### 3.1 .NET WPF のシステムトレイ

```csharp
// WPF — NotifyIcon を使ったシステムトレイ
// NuGet: Hardcodet.NotifyIcon.Wpf
using Hardcodet.Wpf.TaskbarNotification;
using System.Windows;

public partial class App : Application
{
    private TaskbarIcon? _trayIcon;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        _trayIcon = new TaskbarIcon
        {
            IconSource = new BitmapImage(
                new Uri("pack://application:,,,/Assets/tray-icon.ico")),
            ToolTipText = "My Application",
        };

        // コンテキストメニューの作成
        var contextMenu = new ContextMenu();
        contextMenu.Items.Add(new MenuItem
        {
            Header = "表示",
            Command = new RelayCommand(() => MainWindow?.Show()),
        });
        contextMenu.Items.Add(new MenuItem
        {
            Header = "設定",
            Command = new RelayCommand(OpenSettings),
        });
        contextMenu.Items.Add(new Separator());
        contextMenu.Items.Add(new MenuItem
        {
            Header = "終了",
            Command = new RelayCommand(() => Shutdown()),
        });

        _trayIcon.ContextMenu = contextMenu;

        // ダブルクリックでウィンドウ表示
        _trayIcon.TrayMouseDoubleClick += (s, _) =>
        {
            MainWindow?.Show();
            MainWindow?.Activate();
        };

        // バルーン通知の表示
        _trayIcon.ShowBalloonTip(
            "アプリ起動",
            "バックグラウンドで実行中です",
            BalloonIcon.Info);
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _trayIcon?.Dispose();
        base.OnExit(e);
    }
}

// ウィンドウの最小化をトレイに隠す動作に変更
public partial class MainWindow : Window
{
    protected override void OnClosing(CancelEventArgs e)
    {
        // 閉じるボタンでウィンドウを隠す（終了しない）
        e.Cancel = true;
        this.Hide();
    }
}
```

### 3.2 WinUI 3 のシステムトレイ

```csharp
// WinUI 3 — H.NotifyIcon を使ったシステムトレイ
// NuGet: H.NotifyIcon.WinUI
using H.NotifyIcon;

public sealed partial class MainWindow : Window
{
    private TaskbarIcon? _trayIcon;

    public MainWindow()
    {
        InitializeComponent();
        SetupTrayIcon();
    }

    private void SetupTrayIcon()
    {
        _trayIcon = new TaskbarIcon
        {
            // アイコンの設定
            Icon = new System.Drawing.Icon("Assets/tray-icon.ico"),
            ToolTipText = "My WinUI App",
        };

        // メニューフライアウト（WinUI 3 スタイル）
        var flyout = new MenuFlyout();

        var showItem = new MenuFlyoutItem { Text = "表示" };
        showItem.Click += (_, _) =>
        {
            this.Activate();
            // ウィンドウを前面に持ってくる
            var hwnd = WindowNative.GetWindowHandle(this);
            SetForegroundWindow(hwnd);
        };
        flyout.Items.Add(showItem);

        flyout.Items.Add(new MenuFlyoutSeparator());

        var exitItem = new MenuFlyoutItem { Text = "終了" };
        exitItem.Click += (_, _) =>
        {
            _trayIcon?.Dispose();
            this.Close();
        };
        flyout.Items.Add(exitItem);

        _trayIcon.ContextFlyout = flyout;

        // ダブルクリック
        _trayIcon.TrayMouseDoubleClick += (_, _) => this.Activate();
    }

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr hWnd);
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

### 4.1 .NET のグローバルホットキー

```csharp
// Win32 API を使ったグローバルホットキー登録
using System.Runtime.InteropServices;
using System.Windows.Interop;

public class GlobalHotKey : IDisposable
{
    [DllImport("user32.dll")]
    private static extern bool RegisterHotKey(
        IntPtr hWnd, int id, uint fsModifiers, uint vk);

    [DllImport("user32.dll")]
    private static extern bool UnregisterHotKey(IntPtr hWnd, int id);

    private const int WM_HOTKEY = 0x0312;

    // 修飾キー定数
    public const uint MOD_ALT = 0x0001;
    public const uint MOD_CONTROL = 0x0002;
    public const uint MOD_SHIFT = 0x0004;
    public const uint MOD_WIN = 0x0008;
    public const uint MOD_NOREPEAT = 0x4000;

    private readonly IntPtr _hwnd;
    private readonly Dictionary<int, Action> _hotkeys = new();
    private int _nextId = 1;
    private HwndSource? _source;

    public GlobalHotKey(Window window)
    {
        var interopHelper = new WindowInteropHelper(window);
        _hwnd = interopHelper.Handle;

        // メッセージフックを設定
        _source = HwndSource.FromHwnd(_hwnd);
        _source?.AddHook(WndProc);
    }

    /// <summary>
    /// グローバルホットキーを登録する
    /// </summary>
    public int Register(uint modifiers, uint key, Action callback)
    {
        var id = _nextId++;
        if (!RegisterHotKey(_hwnd, id, modifiers | MOD_NOREPEAT, key))
        {
            throw new InvalidOperationException(
                $"Failed to register hotkey (error: {Marshal.GetLastWin32Error()})");
        }
        _hotkeys[id] = callback;
        return id;
    }

    /// <summary>
    /// 特定のホットキーを解除する
    /// </summary>
    public void Unregister(int id)
    {
        UnregisterHotKey(_hwnd, id);
        _hotkeys.Remove(id);
    }

    private IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam,
        IntPtr lParam, ref bool handled)
    {
        if (msg == WM_HOTKEY)
        {
            int id = wParam.ToInt32();
            if (_hotkeys.TryGetValue(id, out var callback))
            {
                callback();
                handled = true;
            }
        }
        return IntPtr.Zero;
    }

    public void Dispose()
    {
        foreach (var id in _hotkeys.Keys.ToList())
        {
            UnregisterHotKey(_hwnd, id);
        }
        _hotkeys.Clear();
        _source?.RemoveHook(WndProc);
    }
}

// 使用例
public partial class MainWindow : Window
{
    private GlobalHotKey? _hotkey;

    protected override void OnSourceInitialized(EventArgs e)
    {
        base.OnSourceInitialized(e);

        _hotkey = new GlobalHotKey(this);

        // Ctrl+Shift+Space で表示/非表示を切り替え
        _hotkey.Register(
            GlobalHotKey.MOD_CONTROL | GlobalHotKey.MOD_SHIFT,
            0x20, // VK_SPACE
            () =>
            {
                if (IsVisible)
                {
                    Hide();
                }
                else
                {
                    Show();
                    Activate();
                }
            });

        // Ctrl+Alt+N で新規作成
        _hotkey.Register(
            GlobalHotKey.MOD_CONTROL | GlobalHotKey.MOD_ALT,
            0x4E, // VK_N
            () =>
            {
                Show();
                Activate();
                CreateNewDocument();
            });
    }

    protected override void OnClosed(EventArgs e)
    {
        _hotkey?.Dispose();
        base.OnClosed(e);
    }
}
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

### 5.1 .NET の自動起動設定

```csharp
// レジストリを使った自動起動設定（Windows）
using Microsoft.Win32;

public class AutoStartService
{
    private const string RunKey = @"SOFTWARE\Microsoft\Windows\CurrentVersion\Run";
    private readonly string _appName;
    private readonly string _appPath;

    public AutoStartService(string appName)
    {
        _appName = appName;
        _appPath = Environment.ProcessPath
            ?? throw new InvalidOperationException("Cannot determine process path");
    }

    /// <summary>
    /// 自動起動を有効/無効にする
    /// </summary>
    public void SetAutoStart(bool enabled)
    {
        using var key = Registry.CurrentUser.OpenSubKey(RunKey, writable: true);
        if (key is null) return;

        if (enabled)
        {
            key.SetValue(_appName, $"\"{_appPath}\" --minimized");
        }
        else
        {
            key.DeleteValue(_appName, throwOnMissingValue: false);
        }
    }

    /// <summary>
    /// 自動起動が有効かどうかを取得
    /// </summary>
    public bool IsAutoStartEnabled()
    {
        using var key = Registry.CurrentUser.OpenSubKey(RunKey);
        return key?.GetValue(_appName) is not null;
    }
}

// タスクスケジューラを使った自動起動（管理者権限不要、より高度な制御）
using System.Diagnostics;

public class TaskSchedulerAutoStart
{
    private readonly string _taskName;
    private readonly string _appPath;

    public TaskSchedulerAutoStart(string taskName)
    {
        _taskName = taskName;
        _appPath = Environment.ProcessPath!;
    }

    /// <summary>
    /// ログオン時に実行するタスクを登録
    /// </summary>
    public void Register()
    {
        // schtasks コマンドでタスクを作成
        var args = $"/create /tn \"{_taskName}\" " +
                   $"/tr \"\\\"{_appPath}\\\" --minimized\" " +
                   "/sc onlogon /rl limited /f";

        var process = Process.Start(new ProcessStartInfo
        {
            FileName = "schtasks.exe",
            Arguments = args,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        });

        process?.WaitForExit();
    }

    /// <summary>
    /// タスクを削除
    /// </summary>
    public void Unregister()
    {
        var process = Process.Start(new ProcessStartInfo
        {
            FileName = "schtasks.exe",
            Arguments = $"/delete /tn \"{_taskName}\" /f",
            UseShellExecute = false,
            CreateNoWindow = true,
        });
        process?.WaitForExit();
    }

    /// <summary>
    /// タスクが登録されているかチェック
    /// </summary>
    public bool IsRegistered()
    {
        var process = Process.Start(new ProcessStartInfo
        {
            FileName = "schtasks.exe",
            Arguments = $"/query /tn \"{_taskName}\"",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        });

        process?.WaitForExit();
        return process?.ExitCode == 0;
    }
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

### 6.1 .NET のクリップボード操作

```csharp
// WPF — クリップボード操作
using System.Windows;

public class ClipboardService : IClipboardService
{
    /// <summary>
    /// テキストをクリップボードにコピー
    /// </summary>
    public void CopyText(string text)
    {
        Clipboard.SetText(text);
    }

    /// <summary>
    /// クリップボードからテキストを取得
    /// </summary>
    public string? PasteText()
    {
        return Clipboard.ContainsText() ? Clipboard.GetText() : null;
    }

    /// <summary>
    /// 画像をクリップボードにコピー
    /// </summary>
    public void CopyImage(BitmapSource image)
    {
        Clipboard.SetImage(image);
    }

    /// <summary>
    /// クリップボードから画像を取得
    /// </summary>
    public BitmapSource? PasteImage()
    {
        return Clipboard.ContainsImage() ? Clipboard.GetImage() : null;
    }

    /// <summary>
    /// ファイルパスをクリップボードにコピー（エクスプローラのコピーと同等）
    /// </summary>
    public void CopyFiles(IEnumerable<string> filePaths)
    {
        var collection = new System.Collections.Specialized.StringCollection();
        foreach (var path in filePaths)
        {
            collection.Add(path);
        }
        Clipboard.SetFileDropList(collection);
    }

    /// <summary>
    /// 複数形式のデータをクリップボードに設定
    /// </summary>
    public void CopyRichContent(string plainText, string htmlText)
    {
        var dataObject = new DataObject();
        dataObject.SetText(plainText, TextDataFormat.UnicodeText);
        dataObject.SetText(htmlText, TextDataFormat.Html);
        Clipboard.SetDataObject(dataObject, copy: true);
    }

    /// <summary>
    /// クリップボードの変更を監視
    /// </summary>
    public void StartMonitoring(Action onClipboardChanged)
    {
        // Win32 API でクリップボードの変更を監視
        // AddClipboardFormatListener を使用
        ClipboardMonitor.Start(onClipboardChanged);
    }
}
```

### 6.2 .NET のドラッグ&ドロップ

```xml
<!-- WPF — ドラッグ&ドロップ対応の XAML -->
<Border
    AllowDrop="True"
    Drop="OnDrop"
    DragEnter="OnDragEnter"
    DragLeave="OnDragLeave"
    BorderBrush="{Binding DropBorderBrush}"
    BorderThickness="2"
    BorderDashStyle="Dash"
    Padding="40">
    <TextBlock Text="ファイルをここにドロップ"
               HorizontalAlignment="Center"
               VerticalAlignment="Center" />
</Border>
```

```csharp
// WPF — ドラッグ&ドロップのコードビハインド
public partial class DropZoneControl : UserControl
{
    public DropZoneControl()
    {
        InitializeComponent();
        AllowDrop = true;
    }

    private void OnDragEnter(object sender, DragEventArgs e)
    {
        if (e.Data.GetDataPresent(DataFormats.FileDrop))
        {
            e.Effects = DragDropEffects.Copy;
            // ドロップゾーンのハイライト
            DropBorder.BorderBrush = Brushes.DodgerBlue;
            DropBorder.Background = new SolidColorBrush(
                Color.FromArgb(30, 30, 144, 255));
        }
        else
        {
            e.Effects = DragDropEffects.None;
        }
        e.Handled = true;
    }

    private void OnDragLeave(object sender, DragEventArgs e)
    {
        DropBorder.BorderBrush = Brushes.Gray;
        DropBorder.Background = Brushes.Transparent;
    }

    private async void OnDrop(object sender, DragEventArgs e)
    {
        DropBorder.BorderBrush = Brushes.Gray;
        DropBorder.Background = Brushes.Transparent;

        if (e.Data.GetDataPresent(DataFormats.FileDrop))
        {
            var files = (string[])e.Data.GetData(DataFormats.FileDrop)!;
            foreach (var filePath in files)
            {
                var fileInfo = new FileInfo(filePath);
                StatusText.Text = $"受信: {fileInfo.Name} ({fileInfo.Length:N0} bytes)";

                // ファイルの処理
                await ProcessDroppedFileAsync(filePath);
            }
        }
    }

    // ドラッグ元の実装（リストからアイテムをドラッグ）
    private void ListItem_MouseMove(object sender, MouseEventArgs e)
    {
        if (e.LeftButton == MouseButtonState.Pressed)
        {
            if (sender is FrameworkElement element &&
                element.DataContext is FileItem item)
            {
                var data = new DataObject(DataFormats.FileDrop,
                    new string[] { item.FullPath });
                DragDrop.DoDragDrop(element, data, DragDropEffects.Copy);
            }
        }
    }
}
```

---

## 7. レジストリ操作

```csharp
// Windows レジストリの読み書き
using Microsoft.Win32;

public class RegistryService
{
    private readonly string _appKey;

    public RegistryService(string appName)
    {
        _appKey = $@"SOFTWARE\{appName}";
    }

    /// <summary>
    /// アプリ設定をレジストリに保存
    /// </summary>
    public void SaveSetting(string name, object value)
    {
        using var key = Registry.CurrentUser.CreateSubKey(_appKey);
        key.SetValue(name, value);
    }

    /// <summary>
    /// アプリ設定をレジストリから読み取り
    /// </summary>
    public T? ReadSetting<T>(string name, T? defaultValue = default)
    {
        using var key = Registry.CurrentUser.OpenSubKey(_appKey);
        var value = key?.GetValue(name);

        if (value is null) return defaultValue;

        return (T)Convert.ChangeType(value, typeof(T));
    }

    /// <summary>
    /// アプリのレジストリキーを全削除
    /// </summary>
    public void DeleteAllSettings()
    {
        Registry.CurrentUser.DeleteSubKeyTree(_appKey, throwOnMissingSubKey: false);
    }

    /// <summary>
    /// ファイル関連付けを登録する
    /// </summary>
    public void RegisterFileAssociation(
        string extension,
        string progId,
        string description,
        string appPath,
        string iconPath)
    {
        // 拡張子の登録
        using (var extKey = Registry.CurrentUser.CreateSubKey(
            $@"SOFTWARE\Classes\{extension}"))
        {
            extKey.SetValue("", progId);
        }

        // ProgID の登録
        using (var progKey = Registry.CurrentUser.CreateSubKey(
            $@"SOFTWARE\Classes\{progId}"))
        {
            progKey.SetValue("", description);

            using (var iconKey = progKey.CreateSubKey("DefaultIcon"))
            {
                iconKey.SetValue("", $"\"{iconPath}\",0");
            }

            using (var commandKey = progKey.CreateSubKey(@"shell\open\command"))
            {
                commandKey.SetValue("", $"\"{appPath}\" \"%1\"");
            }
        }

        // シェルに通知
        SHChangeNotify(0x08000000, 0, IntPtr.Zero, IntPtr.Zero);
    }

    [DllImport("shell32.dll")]
    private static extern void SHChangeNotify(
        int wEventId, int uFlags, IntPtr dwItem1, IntPtr dwItem2);
}
```

---

## 8. シェル統合

```csharp
// エクスプローラのコンテキストメニューに項目を追加
public class ContextMenuRegistration
{
    /// <summary>
    /// 右クリックメニューにアプリのエントリを追加
    /// </summary>
    public static void Register(
        string appName,
        string appPath,
        string menuText,
        string iconPath,
        string[] extensions)
    {
        foreach (var ext in extensions)
        {
            var keyPath = $@"SOFTWARE\Classes\{ext}\shell\{appName}";

            using var key = Registry.CurrentUser.CreateSubKey(keyPath);
            key.SetValue("", menuText);
            key.SetValue("Icon", $"\"{iconPath}\"");

            using var commandKey = key.CreateSubKey("command");
            commandKey.SetValue("", $"\"{appPath}\" \"%1\"");
        }
    }

    /// <summary>
    /// ディレクトリの右クリックメニューに追加
    /// </summary>
    public static void RegisterForDirectories(
        string appName,
        string appPath,
        string menuText)
    {
        var keyPath = $@"SOFTWARE\Classes\Directory\shell\{appName}";
        using var key = Registry.CurrentUser.CreateSubKey(keyPath);
        key.SetValue("", menuText);

        using var commandKey = key.CreateSubKey("command");
        commandKey.SetValue("", $"\"{appPath}\" \"%V\"");

        // 背景の右クリックにも追加
        var bgKeyPath = $@"SOFTWARE\Classes\Directory\Background\shell\{appName}";
        using var bgKey = Registry.CurrentUser.CreateSubKey(bgKeyPath);
        bgKey.SetValue("", menuText);

        using var bgCommandKey = bgKey.CreateSubKey("command");
        bgCommandKey.SetValue("", $"\"{appPath}\" \"%V\"");
    }

    /// <summary>
    /// コンテキストメニューのエントリを削除
    /// </summary>
    public static void Unregister(string appName, string[] extensions)
    {
        foreach (var ext in extensions)
        {
            var keyPath = $@"SOFTWARE\Classes\{ext}\shell\{appName}";
            Registry.CurrentUser.DeleteSubKeyTree(keyPath,
                throwOnMissingSubKey: false);
        }

        Registry.CurrentUser.DeleteSubKeyTree(
            $@"SOFTWARE\Classes\Directory\shell\{appName}",
            throwOnMissingSubKey: false);
        Registry.CurrentUser.DeleteSubKeyTree(
            $@"SOFTWARE\Classes\Directory\Background\shell\{appName}",
            throwOnMissingSubKey: false);
    }
}
```

```csharp
// Windows ジャンプリスト（タスクバーの右クリックメニュー）
// WPF での実装
using System.Windows.Shell;

public class JumpListService
{
    public void SetupJumpList()
    {
        var jumpList = new JumpList();

        // 最近使ったファイルのカテゴリ
        jumpList.ShowRecentCategory = true;

        // カスタムタスク
        jumpList.JumpItems.Add(new JumpTask
        {
            Title = "新規ドキュメント",
            Description = "新しいドキュメントを作成します",
            ApplicationPath = Environment.ProcessPath!,
            Arguments = "--new",
            IconResourcePath = Environment.ProcessPath!,
            IconResourceIndex = 0,
        });

        jumpList.JumpItems.Add(new JumpTask
        {
            Title = "設定を開く",
            Description = "アプリケーション設定を開きます",
            ApplicationPath = Environment.ProcessPath!,
            Arguments = "--settings",
        });

        // カスタムカテゴリ
        jumpList.JumpItems.Add(new JumpTask
        {
            Title = "テンプレート A",
            CustomCategory = "テンプレート",
            ApplicationPath = Environment.ProcessPath!,
            Arguments = "--template A",
        });

        JumpList.SetJumpList(Application.Current, jumpList);
    }

    /// <summary>
    /// 最近使ったファイルをジャンプリストに追加
    /// </summary>
    public void AddRecentFile(string filePath)
    {
        JumpList.AddToRecentCategory(filePath);
    }
}
```

---

## 9. Win32 API の P/Invoke

```csharp
// よく使う Win32 API の P/Invoke 定義集
using System.Runtime.InteropServices;

public static partial class NativeMethods
{
    // ウィンドウを前面に持ってくる
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    // ウィンドウの表示状態を変更
    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    // ウィンドウが最小化されているか
    [DllImport("user32.dll")]
    public static extern bool IsIconic(IntPtr hWnd);

    // フラッシュ（タスクバーでの点滅）
    [DllImport("user32.dll")]
    public static extern bool FlashWindowEx(ref FLASHWINFO pwfi);

    // モニター情報の取得
    [DllImport("user32.dll")]
    public static extern bool GetMonitorInfo(IntPtr hMonitor, ref MONITORINFO lpmi);

    [DllImport("user32.dll")]
    public static extern IntPtr MonitorFromWindow(IntPtr hwnd, uint dwFlags);

    // DPI の取得
    [DllImport("shcore.dll")]
    public static extern int GetDpiForMonitor(
        IntPtr hMonitor, int dpiType, out uint dpiX, out uint dpiY);

    // 電源状態の取得
    [DllImport("kernel32.dll")]
    public static extern bool GetSystemPowerStatus(
        out SYSTEM_POWER_STATUS lpSystemPowerStatus);

    // プロセスの優先度設定
    [DllImport("kernel32.dll")]
    public static extern bool SetPriorityClass(IntPtr hProcess, uint dwPriorityClass);

    // ファイルロック状態チェック
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr CreateFile(
        string lpFileName, uint dwDesiredAccess,
        uint dwShareMode, IntPtr lpSecurityAttributes,
        uint dwCreationDisposition, uint dwFlagsAndAttributes,
        IntPtr hTemplateFile);

    // 定数
    public const int SW_SHOW = 5;
    public const int SW_MINIMIZE = 6;
    public const int SW_RESTORE = 9;
    public const uint MONITOR_DEFAULTTONEAREST = 2;
}

[StructLayout(LayoutKind.Sequential)]
public struct FLASHWINFO
{
    public uint cbSize;
    public IntPtr hwnd;
    public uint dwFlags;
    public uint uCount;
    public uint dwTimeout;
}

[StructLayout(LayoutKind.Sequential)]
public struct MONITORINFO
{
    public int cbSize;
    public RECT rcMonitor;
    public RECT rcWork;
    public uint dwFlags;
}

[StructLayout(LayoutKind.Sequential)]
public struct RECT
{
    public int Left, Top, Right, Bottom;
}

[StructLayout(LayoutKind.Sequential)]
public struct SYSTEM_POWER_STATUS
{
    public byte ACLineStatus;
    public byte BatteryFlag;
    public byte BatteryLifePercent;
    public byte SystemStatusFlag;
    public int BatteryLifeTime;
    public int BatteryFullLifeTime;
}
```

```csharp
// P/Invoke の活用例 — ウィンドウの点滅通知
public static class WindowFlasher
{
    private const uint FLASHW_STOP = 0;
    private const uint FLASHW_CAPTION = 1;
    private const uint FLASHW_TRAY = 2;
    private const uint FLASHW_ALL = FLASHW_CAPTION | FLASHW_TRAY;
    private const uint FLASHW_TIMER = 4;
    private const uint FLASHW_TIMERNOFG = 12;

    /// <summary>
    /// タスクバーでウィンドウを点滅させて注意を引く
    /// </summary>
    public static void Flash(IntPtr hwnd, uint count = 5)
    {
        var info = new FLASHWINFO
        {
            cbSize = (uint)Marshal.SizeOf<FLASHWINFO>(),
            hwnd = hwnd,
            dwFlags = FLASHW_ALL | FLASHW_TIMERNOFG,
            uCount = count,
            dwTimeout = 0,
        };

        NativeMethods.FlashWindowEx(ref info);
    }

    /// <summary>
    /// 点滅を停止する
    /// </summary>
    public static void StopFlash(IntPtr hwnd)
    {
        var info = new FLASHWINFO
        {
            cbSize = (uint)Marshal.SizeOf<FLASHWINFO>(),
            hwnd = hwnd,
            dwFlags = FLASHW_STOP,
        };

        NativeMethods.FlashWindowEx(ref info);
    }
}

// 電源状態の監視
public class PowerMonitor
{
    /// <summary>
    /// バッテリー残量を取得
    /// </summary>
    public static int GetBatteryPercentage()
    {
        NativeMethods.GetSystemPowerStatus(out var status);
        return status.BatteryLifePercent;
    }

    /// <summary>
    /// AC 電源に接続されているか
    /// </summary>
    public static bool IsOnAcPower()
    {
        NativeMethods.GetSystemPowerStatus(out var status);
        return status.ACLineStatus == 1;
    }
}
```

---

## 10. 単一インスタンス制御

```csharp
// アプリの多重起動を防止する
using System.Threading;

public class SingleInstanceGuard : IDisposable
{
    private readonly Mutex _mutex;
    private bool _hasHandle;

    public SingleInstanceGuard(string appId)
    {
        _mutex = new Mutex(false, $"Global\\{appId}");
    }

    /// <summary>
    /// 他のインスタンスが実行中でないか確認
    /// </summary>
    public bool TryAcquire()
    {
        try
        {
            _hasHandle = _mutex.WaitOne(0, false);
            return _hasHandle;
        }
        catch (AbandonedMutexException)
        {
            _hasHandle = true;
            return true;
        }
    }

    public void Dispose()
    {
        if (_hasHandle)
        {
            _mutex.ReleaseMutex();
        }
        _mutex.Dispose();
    }
}

// App.xaml.cs での使用
public partial class App : Application
{
    private SingleInstanceGuard? _guard;

    protected override void OnStartup(StartupEventArgs e)
    {
        _guard = new SingleInstanceGuard("com.mycompany.myapp");

        if (!_guard.TryAcquire())
        {
            // 既存インスタンスをアクティブにする
            ActivateExistingInstance();
            Shutdown();
            return;
        }

        base.OnStartup(e);
    }

    private void ActivateExistingInstance()
    {
        // 名前付きパイプで既存インスタンスに通知
        using var client = new NamedPipeClientStream(".", "MyApp-IPC",
            PipeDirection.Out);
        try
        {
            client.Connect(1000);
            using var writer = new StreamWriter(client);
            writer.WriteLine("ACTIVATE");
            // コマンドライン引数も転送
            writer.WriteLine(string.Join("|", Environment.GetCommandLineArgs()));
        }
        catch (TimeoutException)
        {
            MessageBox.Show("アプリケーションは既に実行中です。");
        }
    }
}

// 既存インスタンスのリスナー
public class SingleInstanceListener : IDisposable
{
    private readonly CancellationTokenSource _cts = new();

    public event Action<string[]>? ArgumentsReceived;

    public void Start()
    {
        Task.Run(async () =>
        {
            while (!_cts.Token.IsCancellationRequested)
            {
                using var server = new NamedPipeServerStream("MyApp-IPC",
                    PipeDirection.In, 1);
                await server.WaitForConnectionAsync(_cts.Token);

                using var reader = new StreamReader(server);
                var command = await reader.ReadLineAsync();
                var argsLine = await reader.ReadLineAsync();

                if (command == "ACTIVATE")
                {
                    var args = argsLine?.Split('|') ?? Array.Empty<string>();
                    ArgumentsReceived?.Invoke(args);
                }
            }
        }, _cts.Token);
    }

    public void Dispose()
    {
        _cts.Cancel();
        _cts.Dispose();
    }
}
```

---

## 11. 印刷機能

```csharp
// WPF — 印刷機能の実装
using System.Printing;
using System.Windows.Controls;
using System.Windows.Documents;

public class PrintService
{
    /// <summary>
    /// 印刷ダイアログを表示してドキュメントを印刷
    /// </summary>
    public bool PrintDocument(FlowDocument document, string title)
    {
        var printDialog = new PrintDialog();

        if (printDialog.ShowDialog() != true)
            return false;

        // FlowDocument を DocumentPaginator に変換
        var paginator = ((IDocumentPaginatorSource)document)
            .DocumentPaginator;

        // ページサイズを設定
        paginator.PageSize = new Size(
            printDialog.PrintableAreaWidth,
            printDialog.PrintableAreaHeight);

        printDialog.PrintDocument(paginator, title);
        return true;
    }

    /// <summary>
    /// ビジュアル要素をそのまま印刷
    /// </summary>
    public bool PrintVisual(Visual visual, string title)
    {
        var printDialog = new PrintDialog();

        if (printDialog.ShowDialog() != true)
            return false;

        printDialog.PrintVisual(visual, title);
        return true;
    }

    /// <summary>
    /// 利用可能なプリンター一覧を取得
    /// </summary>
    public IReadOnlyList<string> GetAvailablePrinters()
    {
        var server = new PrintServer();
        return server.GetPrintQueues()
            .Select(q => q.FullName)
            .ToList();
    }
}
```

---

## FAQ

### Q1: ファイルアクセスのセキュリティは？
メインプロセスでパス検証を必ず行う。ユーザーが選択したパス以外へのアクセスは拒否する。Tauri は capabilities で制御。.NET アプリでは Environment.SpecialFolder を使って安全なパスを取得する。

### Q2: macOS と Windows で通知の動作は違う？
macOS は Notification Center 経由、Windows は Action Center 経由。アイコンサイズやアクションボタンの仕様が異なる。WinUI 3 の AppNotificationManager は Windows 10/11 のトースト通知をフルサポートする。

### Q3: トレイアイコンの推奨サイズは？
macOS: 16x16〜22x22（@2x 対応）、Windows: 16x16〜32x32。Template Image（macOS）を使うとダークモード対応。WPF/WinUI 3 では .ico ファイルを使用する。

### Q4: P/Invoke は .NET 8 以降でも使えるか？
使える。さらに LibraryImport 属性（Source Generator ベース）が推奨されている。DllImport より型安全で高速。

### Q5: Windows サービスとデスクトップアプリを連携させるには？
名前付きパイプ、TCP/IP ソケット、またはメモリマップドファイルで通信する。Windows サービスは Session 0 で動作するため、UI を直接操作できない点に注意。

### Q6: ファイル関連付けは MSIX パッケージでも設定できるか？
はい。Package.appxmanifest の uap:FileTypeAssociation 要素で宣言的に設定できる。レジストリ操作は不要で、アンインストール時に自動的にクリーンアップされる。

### Q7: 多重起動防止は Mutex 以外の方法はあるか？
名前付きパイプ、ファイルロック、またはローカル TCP ポートのバインドでも実現できる。Mutex が最も軽量でシンプル。MSIX パッケージの場合は AppInstance.FindOrRegisterForKey() を使用できる。

---

## まとめ

| 機能 | Electron | Tauri | WPF/WinUI 3 |
|------|----------|-------|-------------|
| ファイルダイアログ | dialog.showOpenDialog | @tauri-apps/plugin-dialog | OpenFileDialog / FileOpenPicker |
| 通知 | Notification | @tauri-apps/plugin-notification | AppNotificationManager |
| トレイ | Tray | TrayIcon | NotifyIcon / H.NotifyIcon |
| ショートカット | globalShortcut | @tauri-apps/plugin-global-shortcut | RegisterHotKey (P/Invoke) |
| 自動起動 | app.setLoginItemSettings | @tauri-apps/plugin-autostart | レジストリ / タスクスケジューラ |
| クリップボード | clipboard | @tauri-apps/plugin-clipboard | System.Windows.Clipboard |
| ドラッグ&ドロップ | HTML5 DnD API | HTML5 DnD API | WPF DragDrop |
| ファイル監視 | chokidar | notify (Rust) | FileSystemWatcher |
| レジストリ | N/A | N/A | Microsoft.Win32.Registry |
| 印刷 | webContents.print() | N/A | PrintDialog |
| 単一インスタンス | app.requestSingleInstanceLock() | N/A | Mutex / NamedPipe |

---

## 次に読むべきガイド
→ [[03-cross-platform.md]] — クロスプラットフォーム対応

---

## 参考文献
1. Electron. "Native File Dialogs." electronjs.org/docs, 2024.
2. Electron. "Tray." electronjs.org/docs/api/tray, 2024.
3. Tauri. "Plugins." tauri.app/plugin, 2024.
4. Microsoft. "Windows App SDK — App Notifications." learn.microsoft.com/windows/apps/windows-app-sdk/notifications, 2024.
5. Microsoft. "P/Invoke in .NET." learn.microsoft.com/dotnet/standard/native-interop/pinvoke, 2024.
6. Microsoft. "JumpList Class." learn.microsoft.com/dotnet/api/system.windows.shell.jumplist, 2024.
7. Microsoft. "File System Watcher." learn.microsoft.com/dotnet/api/system.io.filesystemwatcher, 2024.
8. Microsoft. "Windows Registry." learn.microsoft.com/dotnet/api/microsoft.win32.registry, 2024.

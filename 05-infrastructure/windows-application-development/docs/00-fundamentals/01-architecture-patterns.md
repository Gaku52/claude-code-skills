# アーキテクチャパターン

> デスクトップアプリのアーキテクチャはプロセス分離とIPC通信が核心。メインプロセス/レンダラーモデル、セキュアなIPC設計、preloadスクリプト、コンテキスト分離まで、安全で堅牢なアプリ設計を解説する。さらに、.NET デスクトップにおける MVVM・クリーンアーキテクチャ・DI コンテナ構成、Win32 アプリのメッセージループ設計、マルチウィンドウ管理パターンまで包括的にカバーする。

## この章で学ぶこと

- [ ] メインプロセス/レンダラープロセスモデルを理解する
- [ ] IPC通信パターン（invoke/handle、send/on）を実装できる
- [ ] preloadスクリプトでセキュアなブリッジを構築できる
- [ ] .NET デスクトップアプリケーションのアーキテクチャ層を設計できる
- [ ] Win32 メッセージループの仕組みを理解する
- [ ] マルチウィンドウ管理とプラグインアーキテクチャを構築できる
- [ ] 依存性注入（DI）コンテナを活用したテスタブルな設計ができる

---

## 1. プロセスモデル

```
デスクトップアプリのプロセス構造:

  ┌─────────────────────────────────────────────┐
  │              メインプロセス                    │
  │  (Node.js / Rust バックエンド)                │
  │                                              │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
  │  │ファイルI/O │  │ネイティブ │  │ OS API   │   │
  │  │          │  │メニュー  │  │ 通知/トレイ│   │
  │  └──────────┘  └──────────┘  └──────────┘   │
  │              ▲  IPC  ▼                       │
  ├──────────────┼──────┼───────────────────────┤
  │              ▼      ▲                        │
  │         ┌───────────────┐                    │
  │         │ preload.js    │ ← ブリッジ層       │
  │         │ contextBridge │                    │
  │         └───────┬───────┘                    │
  │                 │                            │
  │  ┌──────────────▼──────────────┐             │
  │  │     レンダラープロセス        │             │
  │  │  (Chromium / WebView)       │             │
  │  │  React / Vue / Svelte       │             │
  │  │  HTML / CSS / JavaScript    │             │
  │  └─────────────────────────────┘             │
  └─────────────────────────────────────────────┘

  Electron のプロセスモデル:
    メインプロセス:   1つ（Node.js ランタイム）
    レンダラープロセス: ウィンドウごとに1つ（Chromium）
    preload:         レンダラーごとに1つ（隔離されたコンテキスト）

  Tauri のプロセスモデル:
    コアプロセス:     1つ（Rust バックエンド）
    WebView プロセス: ウィンドウごとに1つ（OS WebView）
    → Chromium を同梱しないため軽量

  .NET デスクトップ (WPF/WinUI 3) のプロセスモデル:
    単一プロセス:     UI スレッド + バックグラウンドスレッド
    UI スレッド:      メッセージループ（Dispatcher）で UI を管理
    ワーカースレッド:  Task / ThreadPool で非同期処理
```

### 1.1 Electron のメインプロセス

```typescript
// main.ts — Electron メインプロセス
import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'path';

let mainWindow: BrowserWindow | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      // セキュリティ設定
      nodeIntegration: false,      // レンダラーで Node.js 無効
      contextIsolation: true,      // コンテキスト分離有効
      sandbox: true,               // サンドボックス有効
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  // 開発時: Vite dev server
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
```

### 1.2 Tauri のコアプロセス

```rust
// src-tauri/src/main.rs — Tauri コアプロセス
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! From Rust backend.", name)
}

#[tauri::command]
async fn read_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path)
        .map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet, read_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 1.3 .NET デスクトップのプロセスモデル

```
.NET デスクトップアプリのスレッドモデル:

  ┌──────────────────────────────────────────────┐
  │              アプリケーション                   │
  │                                               │
  │  ┌─────────────────────────────────────────┐  │
  │  │           UI スレッド (STA)               │  │
  │  │  ┌─────────┐  ┌──────────┐  ┌────────┐ │  │
  │  │  │DispatcherQueue│  │ XAML    │  │メッセージ│ │  │
  │  │  │ メッセージループ│  │ レンダリング│  │ ポンプ │ │  │
  │  │  └─────────┘  └──────────┘  └────────┘ │  │
  │  └──────────────┬──────────────────────────┘  │
  │                 │ Dispatcher.Invoke            │
  │                 │ DispatcherQueue.TryEnqueue    │
  │  ┌──────────────▼──────────────────────────┐  │
  │  │        バックグラウンドスレッド              │  │
  │  │  ┌─────────┐  ┌──────────┐  ┌────────┐ │  │
  │  │  │ Task     │  │ ThreadPool│  │ Timer  │ │  │
  │  │  │ async/await│  │ WorkItem │  │        │ │  │
  │  │  └─────────┘  └──────────┘  └────────┘ │  │
  │  └─────────────────────────────────────────┘  │
  └──────────────────────────────────────────────┘
```

```csharp
// WPF の UI スレッドとバックグラウンド処理
using System.Windows;
using System.Windows.Threading;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    private async void LoadDataButton_Click(object sender, RoutedEventArgs e)
    {
        // UI スレッドでボタンを無効化
        LoadDataButton.IsEnabled = false;
        StatusText.Text = "読み込み中...";

        try
        {
            // バックグラウンドスレッドで重い処理を実行
            var data = await Task.Run(() =>
            {
                // CPU バウンドな処理（別スレッドで実行される）
                Thread.Sleep(3000); // シミュレーション
                return LoadExpensiveData();
            });

            // await の後は自動的に UI スレッドに戻る
            StatusText.Text = $"完了: {data.Count} 件取得";
            DataGrid.ItemsSource = data;
        }
        catch (Exception ex)
        {
            StatusText.Text = $"エラー: {ex.Message}";
        }
        finally
        {
            LoadDataButton.IsEnabled = true;
        }
    }

    // Dispatcher を使った明示的な UI スレッドへのマーシャリング
    private void BackgroundWorker_DoWork()
    {
        for (int i = 0; i < 100; i++)
        {
            Thread.Sleep(50);
            // UI スレッドで進捗を更新
            Dispatcher.Invoke(() =>
            {
                ProgressBar.Value = i + 1;
            });
        }
    }
}
```

```csharp
// WinUI 3 の DispatcherQueue を使ったスレッド管理
using Microsoft.UI.Dispatching;

public sealed partial class MainPage : Page
{
    private readonly DispatcherQueue _dispatcherQueue;

    public MainPage()
    {
        InitializeComponent();
        _dispatcherQueue = DispatcherQueue.GetForCurrentThread();
    }

    private void StartBackgroundWork()
    {
        Task.Run(() =>
        {
            // バックグラウンド処理
            var result = PerformHeavyComputation();

            // UI スレッドに結果を返す
            _dispatcherQueue.TryEnqueue(() =>
            {
                ResultText.Text = result.ToString();
            });
        });
    }

    // DispatcherQueue のプライオリティ指定
    private void UpdateUIWithPriority(string message,
        DispatcherQueuePriority priority = DispatcherQueuePriority.Normal)
    {
        _dispatcherQueue.TryEnqueue(priority, () =>
        {
            StatusText.Text = message;
        });
    }
}
```

---

## 2. IPC 通信パターン

```
IPC（Inter-Process Communication）パターン:

  パターン1: Request-Response（invoke/handle）
  ┌──────────┐  invoke('get-data', args)  ┌──────────┐
  │レンダラー  │ ───────────────────────→  │メイン     │
  │          │                            │          │
  │          │  ←──────────────────────── │          │
  │          │  Promise<result>           │          │
  └──────────┘                            └──────────┘

  パターン2: Fire-and-Forget（send/on）
  ┌──────────┐  send('log', data)         ┌──────────┐
  │レンダラー  │ ───────────────────────→  │メイン     │
  │          │  （応答なし）               │          │
  └──────────┘                            └──────────┘

  パターン3: Push（メイン→レンダラー）
  ┌──────────┐                            ┌──────────┐
  │レンダラー  │  ←──────────────────────  │メイン     │
  │          │  webContents.send('event') │          │
  └──────────┘                            └──────────┘

  パターン4: Bidirectional Stream（MessagePort）
  ┌──────────┐  port.postMessage(data)    ┌──────────┐
  │レンダラー  │ ←───────────────────────→ │メイン     │
  │          │  port.onmessage            │          │
  └──────────┘                            └──────────┘
```

### 2.1 Electron IPC 実装

```typescript
// preload.ts — セキュアなブリッジ
import { contextBridge, ipcRenderer } from 'electron';

// レンダラーに公開するAPI（ホワイトリスト方式）
contextBridge.exposeInMainWorld('electronAPI', {
  // パターン1: Request-Response
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  readFile: (path: string) => ipcRenderer.invoke('read-file', path),
  saveFile: (path: string, data: string) =>
    ipcRenderer.invoke('save-file', path, data),

  // パターン2: Fire-and-Forget
  logEvent: (event: string) => ipcRenderer.send('log-event', event),

  // パターン3: メインからの通知を受信
  onUpdateAvailable: (callback: (version: string) => void) => {
    const handler = (_event: any, version: string) => callback(version);
    ipcRenderer.on('update-available', handler);
    // クリーンアップ関数を返す
    return () => ipcRenderer.removeListener('update-available', handler);
  },
});

// main.ts — ハンドラー登録
ipcMain.handle('get-app-version', () => {
  return app.getVersion();
});

ipcMain.handle('read-file', async (_event, filePath: string) => {
  // パス検証（セキュリティ）
  const safePath = path.resolve(filePath);
  if (!safePath.startsWith(app.getPath('documents'))) {
    throw new Error('Access denied: path outside documents');
  }
  return fs.promises.readFile(safePath, 'utf-8');
});

ipcMain.handle('save-file', async (_event, filePath: string, data: string) => {
  const safePath = path.resolve(filePath);
  if (!safePath.startsWith(app.getPath('documents'))) {
    throw new Error('Access denied');
  }
  await fs.promises.writeFile(safePath, data, 'utf-8');
  return { success: true };
});

// メイン→レンダラー通知
function notifyUpdate(version: string) {
  mainWindow?.webContents.send('update-available', version);
}
```

### 2.2 Tauri コマンド通信

```typescript
// フロントエンド（TypeScript）
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

// コマンド呼び出し（Request-Response）
const greeting = await invoke<string>('greet', { name: 'Gaku' });

// イベント受信（メイン→フロントエンド）
const unlisten = await listen<string>('file-changed', (event) => {
  console.log('File changed:', event.payload);
});

// クリーンアップ
unlisten();
```

### 2.3 MessagePort による高速双方向通信

```typescript
// main.ts — MessagePort の作成と転送
import { MessageChannelMain } from 'electron';

function setupMessagePort() {
  const { port1, port2 } = new MessageChannelMain();

  // メインプロセス側でポートを使用
  port1.on('message', (event) => {
    console.log('Received from renderer:', event.data);
    // ストリーミングデータの処理
    if (event.data.type === 'audio-chunk') {
      processAudioChunk(event.data.buffer);
    }
  });
  port1.start();

  // レンダラーにポートを転送
  mainWindow.webContents.postMessage('port-transfer', null, [port2]);
}

// preload.ts — MessagePort の受信
ipcRenderer.on('port-transfer', (event) => {
  const port = event.ports[0];
  contextBridge.exposeInMainWorld('dataChannel', {
    send: (data: any) => port.postMessage(data),
    onMessage: (callback: (data: any) => void) => {
      port.onmessage = (event) => callback(event.data);
    },
  });
});
```

### 2.4 SharedArrayBuffer による共有メモリ通信

```typescript
// main.ts — SharedArrayBuffer を使った高性能データ共有
// 注意: CSP で cross-origin-opener-policy と
//       cross-origin-embedder-policy の設定が必要

function setupSharedMemory() {
  // 共有バッファを作成（1MB）
  const sharedBuffer = new SharedArrayBuffer(1024 * 1024);
  const view = new Int32Array(sharedBuffer);

  // メインプロセスでデータを書き込み
  Atomics.store(view, 0, 42);

  // レンダラーに SharedArrayBuffer を送信
  mainWindow.webContents.send('shared-buffer', sharedBuffer);
}

// renderer — SharedArrayBuffer の利用
window.electronAPI.onSharedBuffer((buffer: SharedArrayBuffer) => {
  const view = new Int32Array(buffer);
  // Atomics API でスレッドセーフにアクセス
  const value = Atomics.load(view, 0);
  console.log('Shared value:', value); // 42

  // 値の更新（他のスレッドにも即座に反映）
  Atomics.store(view, 0, 100);
});
```

### 2.5 .NET アプリケーション内の通信パターン

```csharp
// Messenger パターン（CommunityToolkit.Mvvm）
// ViewModel 間の疎結合な通信を実現する

// メッセージの定義
public sealed class NavigationMessage : ValueChangedMessage<string>
{
    public NavigationMessage(string pageName) : base(pageName) { }
}

public sealed class DataLoadedMessage
{
    public List<Customer> Customers { get; init; } = new();
    public DateTime LoadedAt { get; init; } = DateTime.Now;
}

// 送信側 ViewModel
public partial class SidebarViewModel : ObservableRecipient
{
    [RelayCommand]
    private void NavigateTo(string pageName)
    {
        // メッセージを送信
        Messenger.Send(new NavigationMessage(pageName));
    }
}

// 受信側 ViewModel
public partial class ShellViewModel : ObservableRecipient,
    IRecipient<NavigationMessage>
{
    public ShellViewModel()
    {
        // メッセンジャーに登録（IsActive = true で自動登録）
        IsActive = true;
    }

    public void Receive(NavigationMessage message)
    {
        // メッセージを受信して処理
        CurrentPage = message.Value switch
        {
            "Home" => new HomeViewModel(),
            "Settings" => new SettingsViewModel(),
            _ => throw new ArgumentException($"Unknown page: {message.Value}")
        };
    }
}
```

```csharp
// EventAggregator パターン（Prism フレームワーク）
using Prism.Events;

// イベントの定義
public class OrderCreatedEvent : PubSubEvent<Order> { }
public class CustomerSelectedEvent : PubSubEvent<Customer> { }

// パブリッシャー
public class OrderViewModel
{
    private readonly IEventAggregator _eventAggregator;

    public OrderViewModel(IEventAggregator eventAggregator)
    {
        _eventAggregator = eventAggregator;
    }

    private void CreateOrder()
    {
        var order = new Order { /* ... */ };
        // イベントを発行
        _eventAggregator.GetEvent<OrderCreatedEvent>().Publish(order);
    }
}

// サブスクライバー
public class DashboardViewModel
{
    public DashboardViewModel(IEventAggregator eventAggregator)
    {
        // イベントを購読
        eventAggregator.GetEvent<OrderCreatedEvent>()
            .Subscribe(OnOrderCreated,
                ThreadOption.UIThread,       // UI スレッドで実行
                keepSubscriberReferenceAlive: false,  // 弱参照
                filter: order => order.Amount > 1000); // フィルタ条件
    }

    private void OnOrderCreated(Order order)
    {
        // 注文作成時の処理
        TotalOrders++;
        RecentOrders.Insert(0, order);
    }
}
```

---

## 3. セキュリティモデル

```
セキュリティ多層防御:

  レイヤー1: プロセス分離
    → レンダラーは Node.js API にアクセス不可
    → nodeIntegration: false（必須）

  レイヤー2: コンテキスト分離
    → contextIsolation: true（必須）
    → preload と Web ページは別コンテキスト

  レイヤー3: サンドボックス
    → sandbox: true
    → ファイルシステム・プロセスへの直接アクセス不可

  レイヤー4: CSP（Content Security Policy）
    → インラインスクリプト禁止
    → 外部リソース読み込み制限

  レイヤー5: API ホワイトリスト
    → contextBridge で必要な API のみ公開
    → 入力検証をメインプロセス側で実施

  Tauri のセキュリティモデル:
    → Capabilities（権限宣言）でAPI単位の許可
    → デフォルトで全API無効
    → ウィンドウ単位で権限を設定可能

  .NET デスクトップのセキュリティモデル:
    → CAS (Code Access Security) は .NET Core 以降廃止
    → MSIX パッケージでサンドボックス配布可能
    → Windows Defender Application Control (WDAC) 対応
    → コード署名による改ざん防止
```

```typescript
// Electron — CSP 設定
// main.ts
mainWindow.webContents.session.webRequest.onHeadersReceived(
  (details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self';" +
          "script-src 'self';" +
          "style-src 'self' 'unsafe-inline';" +
          "img-src 'self' data: https:;" +
          "connect-src 'self' https://api.example.com;"
        ],
      },
    });
  }
);
```

```json
// Tauri — capabilities 設定
// src-tauri/capabilities/default.json
{
  "identifier": "default",
  "description": "Default capabilities",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "dialog:allow-open",
    "dialog:allow-save",
    "fs:allow-read",
    "notification:default"
  ]
}
```

### 3.1 Tauri のスコープ付きファイルアクセス

```json
// src-tauri/capabilities/file-access.json
{
  "identifier": "file-access",
  "description": "Scoped file system access",
  "windows": ["main"],
  "permissions": [
    {
      "identifier": "fs:allow-read",
      "allow": [
        { "path": "$DOCUMENT/**" },
        { "path": "$APPDATA/**" }
      ],
      "deny": [
        { "path": "$DOCUMENT/secret/**" }
      ]
    },
    {
      "identifier": "fs:allow-write",
      "allow": [
        { "path": "$APPDATA/**" }
      ]
    }
  ]
}
```

```rust
// src-tauri/src/security.rs — 入力検証とサニタイズ
use std::path::{Path, PathBuf};
use tauri::AppHandle;

/// パストラバーサル攻撃を防止するパス検証
pub fn validate_path(
    app: &AppHandle,
    requested_path: &str,
) -> Result<PathBuf, String> {
    let base_dir = app
        .path()
        .document_dir()
        .map_err(|e| format!("Failed to get document dir: {}", e))?;

    let resolved = base_dir.join(requested_path);
    let canonical = resolved
        .canonicalize()
        .map_err(|e| format!("Invalid path: {}", e))?;

    // 正規化されたパスがベースディレクトリ内にあることを確認
    if !canonical.starts_with(&base_dir) {
        return Err("Access denied: path traversal detected".to_string());
    }

    Ok(canonical)
}

/// ファイル名のサニタイズ
pub fn sanitize_filename(name: &str) -> String {
    name.chars()
        .filter(|c| !matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|'))
        .collect::<String>()
        .trim()
        .to_string()
}

#[tauri::command]
pub async fn secure_read_file(
    app: AppHandle,
    path: String,
) -> Result<String, String> {
    let safe_path = validate_path(&app, &path)?;

    // ファイルサイズチェック（100MB 上限）
    let metadata = std::fs::metadata(&safe_path)
        .map_err(|e| format!("Cannot read metadata: {}", e))?;
    if metadata.len() > 100 * 1024 * 1024 {
        return Err("File too large (max 100MB)".to_string());
    }

    std::fs::read_to_string(&safe_path)
        .map_err(|e| format!("Read failed: {}", e))
}
```

### 3.2 .NET デスクトップのセキュリティ実装

```csharp
// セキュアなデータ保存（DPAPI を使用）
using System.Security.Cryptography;
using System.Text;

public class SecureStorage
{
    private readonly string _storagePath;

    public SecureStorage(string appName)
    {
        _storagePath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            appName,
            "secure");
        Directory.CreateDirectory(_storagePath);
    }

    /// <summary>
    /// DPAPI（Data Protection API）で暗号化して保存
    /// 現在のユーザーのみが復号可能
    /// </summary>
    public void SaveSecure(string key, string value)
    {
        var data = Encoding.UTF8.GetBytes(value);
        var encrypted = ProtectedData.Protect(
            data,
            entropy: Encoding.UTF8.GetBytes(key),
            scope: DataProtectionScope.CurrentUser);

        var filePath = Path.Combine(_storagePath, SanitizeKey(key));
        File.WriteAllBytes(filePath, encrypted);
    }

    /// <summary>
    /// DPAPI で復号して読み取り
    /// </summary>
    public string? LoadSecure(string key)
    {
        var filePath = Path.Combine(_storagePath, SanitizeKey(key));
        if (!File.Exists(filePath)) return null;

        try
        {
            var encrypted = File.ReadAllBytes(filePath);
            var decrypted = ProtectedData.Unprotect(
                encrypted,
                entropy: Encoding.UTF8.GetBytes(key),
                scope: DataProtectionScope.CurrentUser);
            return Encoding.UTF8.GetString(decrypted);
        }
        catch (CryptographicException)
        {
            // 別ユーザーのデータや改ざんされたデータ
            return null;
        }
    }

    private static string SanitizeKey(string key) =>
        Convert.ToBase64String(SHA256.HashData(Encoding.UTF8.GetBytes(key)))
            .Replace("/", "_")
            .Replace("+", "-");
}
```

```csharp
// Windows Credential Manager を使った認証情報管理
using System.Runtime.InteropServices;

public static class CredentialManager
{
    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern bool CredWrite(ref CREDENTIAL credential, uint flags);

    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern bool CredRead(
        string target,
        CRED_TYPE type,
        uint reservedFlag,
        out IntPtr credentialPtr);

    [DllImport("advapi32.dll")]
    private static extern void CredFree(IntPtr credential);

    public static void Save(string target, string username, string password)
    {
        var credential = new CREDENTIAL
        {
            TargetName = target,
            UserName = username,
            CredentialBlob = Marshal.StringToCoTaskMemUni(password),
            CredentialBlobSize = (uint)(password.Length * 2),
            Type = CRED_TYPE.GENERIC,
            Persist = CRED_PERSIST.LOCAL_MACHINE,
        };

        if (!CredWrite(ref credential, 0))
        {
            throw new InvalidOperationException(
                $"Failed to save credential: {Marshal.GetLastWin32Error()}");
        }
    }

    public static (string Username, string Password)? Load(string target)
    {
        if (!CredRead(target, CRED_TYPE.GENERIC, 0, out var credPtr))
            return null;

        try
        {
            var cred = Marshal.PtrToStructure<CREDENTIAL>(credPtr);
            var password = Marshal.PtrToStringUni(
                cred.CredentialBlob, (int)cred.CredentialBlobSize / 2);
            return (cred.UserName, password ?? "");
        }
        finally
        {
            CredFree(credPtr);
        }
    }
}
```

---

## 4. preload スクリプト設計

```
preload 設計原則:

  ✓ ホワイトリスト方式（必要な API のみ公開）
  ✓ 入力のサニタイズ（レンダラーからの入力を信頼しない）
  ✓ 型安全な API 定義
  ✗ ipcRenderer を直接公開しない
  ✗ require/import を公開しない
  ✗ Node.js API を直接公開しない
```

```typescript
// preload.ts — 型安全な API 設計
import { contextBridge, ipcRenderer } from 'electron';

// API の型定義
export interface ElectronAPI {
  // ファイル操作
  file: {
    open: () => Promise<{ path: string; content: string } | null>;
    save: (content: string) => Promise<boolean>;
    saveAs: (content: string) => Promise<string | null>;
  };
  // アプリ情報
  app: {
    getVersion: () => Promise<string>;
    getPlatform: () => string;
  };
  // イベント
  events: {
    onMenuAction: (callback: (action: string) => void) => () => void;
  };
}

contextBridge.exposeInMainWorld('electronAPI', {
  file: {
    open: () => ipcRenderer.invoke('file:open'),
    save: (content: string) => ipcRenderer.invoke('file:save', content),
    saveAs: (content: string) => ipcRenderer.invoke('file:saveAs', content),
  },
  app: {
    getVersion: () => ipcRenderer.invoke('app:version'),
    getPlatform: () => process.platform,
  },
  events: {
    onMenuAction: (callback: (action: string) => void) => {
      const handler = (_: any, action: string) => callback(action);
      ipcRenderer.on('menu:action', handler);
      return () => ipcRenderer.removeListener('menu:action', handler);
    },
  },
} satisfies ElectronAPI);

// renderer.d.ts — レンダラー側の型定義
declare global {
  interface Window {
    electronAPI: import('./preload').ElectronAPI;
  }
}
```

### 4.1 preload の高度なパターン

```typescript
// preload.ts — バリデーション付き API 設計
import { contextBridge, ipcRenderer } from 'electron';

// 入力バリデーション関数
function validateFilePath(path: string): string {
  if (typeof path !== 'string') throw new Error('Path must be a string');
  if (path.length === 0) throw new Error('Path cannot be empty');
  if (path.length > 32767) throw new Error('Path too long');
  // パストラバーサル防止
  if (path.includes('..')) throw new Error('Path traversal not allowed');
  return path;
}

function validateContent(content: string, maxSize = 10 * 1024 * 1024): string {
  if (typeof content !== 'string') throw new Error('Content must be a string');
  if (content.length > maxSize) throw new Error('Content too large');
  return content;
}

// 率制限（レンダラーからの過剰な呼び出しを防止）
function createRateLimiter(maxCalls: number, windowMs: number) {
  const calls: number[] = [];
  return () => {
    const now = Date.now();
    // ウィンドウ外の古い呼び出しを除去
    while (calls.length > 0 && calls[0]! < now - windowMs) {
      calls.shift();
    }
    if (calls.length >= maxCalls) {
      throw new Error('Rate limit exceeded');
    }
    calls.push(now);
  };
}

const fileOpenLimiter = createRateLimiter(10, 60000); // 1分に10回まで
const fileSaveLimiter = createRateLimiter(5, 60000);  // 1分に5回まで

contextBridge.exposeInMainWorld('electronAPI', {
  file: {
    open: () => {
      fileOpenLimiter();
      return ipcRenderer.invoke('file:open');
    },
    save: (path: string, content: string) => {
      fileSaveLimiter();
      return ipcRenderer.invoke('file:save',
        validateFilePath(path),
        validateContent(content));
    },
    watch: (path: string, callback: (event: string) => void) => {
      validateFilePath(path);
      const handler = (_: any, event: string) => callback(event);
      ipcRenderer.on(`file:changed:${path}`, handler);
      ipcRenderer.send('file:watch', path);
      return () => {
        ipcRenderer.removeListener(`file:changed:${path}`, handler);
        ipcRenderer.send('file:unwatch', path);
      };
    },
  },
});
```

---

## 5. クリーンアーキテクチャ（.NET デスクトップ）

```
クリーンアーキテクチャのレイヤー構成:

  ┌─────────────────────────────────────────────┐
  │          プレゼンテーション層                  │
  │  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
  │  │ View     │  │ ViewModel│  │ Converter │ │
  │  │ (XAML)   │  │          │  │           │ │
  │  └──────────┘  └──────────┘  └───────────┘ │
  ├─────────────────────────────────────────────┤
  │          アプリケーション層                    │
  │  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
  │  │ UseCase  │  │ DTO      │  │ Service   │ │
  │  │ (CQRS)   │  │          │  │ Interface │ │
  │  └──────────┘  └──────────┘  └───────────┘ │
  ├─────────────────────────────────────────────┤
  │          ドメイン層                           │
  │  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
  │  │ Entity   │  │ValueObject│  │ Repository│ │
  │  │          │  │          │  │ Interface │ │
  │  └──────────┘  └──────────┘  └───────────┘ │
  ├─────────────────────────────────────────────┤
  │          インフラストラクチャ層               │
  │  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
  │  │ DB Access│  │ File I/O │  │ HTTP      │ │
  │  │ (EF Core)│  │          │  │ Client    │ │
  │  └──────────┘  └──────────┘  └───────────┘ │
  └─────────────────────────────────────────────┘

  依存性の方向: 外側 → 内側（内側は外側に依存しない）
```

```csharp
// ドメイン層 — エンティティとバリューオブジェクト
namespace MyApp.Domain.Entities;

public class Customer
{
    public CustomerId Id { get; private set; }
    public string Name { get; private set; }
    public Email Email { get; private set; }
    public DateTime CreatedAt { get; private set; }
    public DateTime? UpdatedAt { get; private set; }

    // ファクトリメソッドでバリデーション付き生成
    public static Customer Create(string name, string email)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new DomainException("Name is required");
        if (name.Length > 100)
            throw new DomainException("Name must be 100 chars or less");

        return new Customer
        {
            Id = CustomerId.New(),
            Name = name.Trim(),
            Email = Email.Create(email),
            CreatedAt = DateTime.UtcNow,
        };
    }

    public void UpdateName(string newName)
    {
        if (string.IsNullOrWhiteSpace(newName))
            throw new DomainException("Name is required");
        Name = newName.Trim();
        UpdatedAt = DateTime.UtcNow;
    }
}

// バリューオブジェクト
public record Email
{
    public string Value { get; }

    private Email(string value) => Value = value;

    public static Email Create(string email)
    {
        if (string.IsNullOrWhiteSpace(email))
            throw new DomainException("Email is required");
        if (!email.Contains('@'))
            throw new DomainException("Invalid email format");
        return new Email(email.ToLowerInvariant().Trim());
    }
}

public record CustomerId(Guid Value)
{
    public static CustomerId New() => new(Guid.NewGuid());
}
```

```csharp
// ドメイン層 — リポジトリインターフェース
namespace MyApp.Domain.Repositories;

public interface ICustomerRepository
{
    Task<Customer?> GetByIdAsync(CustomerId id, CancellationToken ct = default);
    Task<IReadOnlyList<Customer>> GetAllAsync(CancellationToken ct = default);
    Task<IReadOnlyList<Customer>> SearchAsync(string query, CancellationToken ct = default);
    Task AddAsync(Customer customer, CancellationToken ct = default);
    Task UpdateAsync(Customer customer, CancellationToken ct = default);
    Task DeleteAsync(CustomerId id, CancellationToken ct = default);
}
```

```csharp
// アプリケーション層 — ユースケース（CQRS パターン）
using MediatR;

namespace MyApp.Application.Customers.Commands;

// コマンド定義
public record CreateCustomerCommand(string Name, string Email) : IRequest<CustomerId>;

// コマンドハンドラー
public class CreateCustomerHandler : IRequestHandler<CreateCustomerCommand, CustomerId>
{
    private readonly ICustomerRepository _repository;
    private readonly IUnitOfWork _unitOfWork;

    public CreateCustomerHandler(
        ICustomerRepository repository,
        IUnitOfWork unitOfWork)
    {
        _repository = repository;
        _unitOfWork = unitOfWork;
    }

    public async Task<CustomerId> Handle(
        CreateCustomerCommand command,
        CancellationToken ct)
    {
        var customer = Customer.Create(command.Name, command.Email);
        await _repository.AddAsync(customer, ct);
        await _unitOfWork.SaveChangesAsync(ct);
        return customer.Id;
    }
}

// クエリ定義
public record GetCustomerByIdQuery(CustomerId Id) : IRequest<CustomerDto?>;

public class GetCustomerByIdHandler : IRequestHandler<GetCustomerByIdQuery, CustomerDto?>
{
    private readonly ICustomerRepository _repository;

    public GetCustomerByIdHandler(ICustomerRepository repository)
    {
        _repository = repository;
    }

    public async Task<CustomerDto?> Handle(
        GetCustomerByIdQuery query,
        CancellationToken ct)
    {
        var customer = await _repository.GetByIdAsync(query.Id, ct);
        return customer is null ? null : CustomerDto.FromEntity(customer);
    }
}

// DTO
public record CustomerDto(Guid Id, string Name, string Email, DateTime CreatedAt)
{
    public static CustomerDto FromEntity(Customer c) =>
        new(c.Id.Value, c.Name, c.Email.Value, c.CreatedAt);
}
```

```csharp
// インフラストラクチャ層 — EF Core リポジトリ実装
using Microsoft.EntityFrameworkCore;

namespace MyApp.Infrastructure.Persistence;

public class CustomerRepository : ICustomerRepository
{
    private readonly AppDbContext _context;

    public CustomerRepository(AppDbContext context)
    {
        _context = context;
    }

    public async Task<Customer?> GetByIdAsync(
        CustomerId id, CancellationToken ct = default)
    {
        return await _context.Customers
            .FirstOrDefaultAsync(c => c.Id == id, ct);
    }

    public async Task<IReadOnlyList<Customer>> GetAllAsync(
        CancellationToken ct = default)
    {
        return await _context.Customers
            .OrderBy(c => c.Name)
            .ToListAsync(ct);
    }

    public async Task<IReadOnlyList<Customer>> SearchAsync(
        string query, CancellationToken ct = default)
    {
        return await _context.Customers
            .Where(c => c.Name.Contains(query) ||
                        c.Email.Value.Contains(query))
            .OrderBy(c => c.Name)
            .ToListAsync(ct);
    }

    public async Task AddAsync(Customer customer, CancellationToken ct = default)
    {
        await _context.Customers.AddAsync(customer, ct);
    }

    public Task UpdateAsync(Customer customer, CancellationToken ct = default)
    {
        _context.Customers.Update(customer);
        return Task.CompletedTask;
    }

    public async Task DeleteAsync(CustomerId id, CancellationToken ct = default)
    {
        var customer = await GetByIdAsync(id, ct);
        if (customer is not null)
            _context.Customers.Remove(customer);
    }
}
```

---

## 6. 依存性注入（DI）の設計

```csharp
// App.xaml.cs — DI コンテナの構成（WinUI 3）
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace MyApp;

public partial class App : Application
{
    public IHost Host { get; }

    public static T GetService<T>() where T : class
    {
        var app = (App)Current;
        return app.Host.Services.GetRequiredService<T>();
    }

    public App()
    {
        InitializeComponent();

        Host = Microsoft.Extensions.Hosting.Host
            .CreateDefaultBuilder()
            .ConfigureServices((context, services) =>
            {
                // ViewModel の登録
                services.AddTransient<MainViewModel>();
                services.AddTransient<SettingsViewModel>();
                services.AddTransient<CustomerListViewModel>();
                services.AddTransient<CustomerDetailViewModel>();

                // View の登録
                services.AddTransient<MainPage>();
                services.AddTransient<SettingsPage>();
                services.AddTransient<CustomerListPage>();

                // サービスの登録
                services.AddSingleton<INavigationService, NavigationService>();
                services.AddSingleton<IDialogService, DialogService>();
                services.AddSingleton<ISettingsService, SettingsService>();

                // リポジトリの登録
                services.AddScoped<ICustomerRepository, CustomerRepository>();
                services.AddScoped<IUnitOfWork, UnitOfWork>();

                // データベースコンテキスト
                services.AddDbContext<AppDbContext>(options =>
                {
                    var dbPath = Path.Combine(
                        Environment.GetFolderPath(
                            Environment.SpecialFolder.LocalApplicationData),
                        "MyApp", "app.db");
                    options.UseSqlite($"Data Source={dbPath}");
                });

                // HTTP クライアント
                services.AddHttpClient<IApiClient, ApiClient>(client =>
                {
                    client.BaseAddress = new Uri("https://api.example.com");
                    client.Timeout = TimeSpan.FromSeconds(30);
                });

                // MediatR（CQRS）
                services.AddMediatR(cfg =>
                {
                    cfg.RegisterServicesFromAssemblyContaining<CreateCustomerCommand>();
                    cfg.AddBehavior(typeof(IPipelineBehavior<,>),
                        typeof(ValidationBehavior<,>));
                    cfg.AddBehavior(typeof(IPipelineBehavior<,>),
                        typeof(LoggingBehavior<,>));
                });

                // ロギング
                services.AddLogging(builder =>
                {
                    builder.AddDebug();
                    builder.AddFile("logs/app-{Date}.log");
                });
            })
            .Build();
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        _window = new MainWindow();
        _window.Activate();
    }
}
```

```csharp
// ViewModel の DI 活用例
public partial class CustomerListViewModel : ObservableObject
{
    private readonly IMediator _mediator;
    private readonly INavigationService _navigation;
    private readonly IDialogService _dialog;

    // コンストラクタインジェクション
    public CustomerListViewModel(
        IMediator mediator,
        INavigationService navigation,
        IDialogService dialog)
    {
        _mediator = mediator;
        _navigation = navigation;
        _dialog = dialog;
    }

    [ObservableProperty]
    private ObservableCollection<CustomerDto> _customers = new();

    [ObservableProperty]
    private string _searchQuery = "";

    [ObservableProperty]
    private bool _isLoading;

    [RelayCommand]
    private async Task LoadCustomersAsync()
    {
        IsLoading = true;
        try
        {
            var result = await _mediator.Send(new GetAllCustomersQuery());
            Customers = new ObservableCollection<CustomerDto>(result);
        }
        finally
        {
            IsLoading = false;
        }
    }

    [RelayCommand]
    private async Task DeleteCustomerAsync(CustomerDto customer)
    {
        var confirmed = await _dialog.ShowConfirmAsync(
            "削除確認",
            $"{customer.Name} を削除しますか？");

        if (confirmed)
        {
            await _mediator.Send(
                new DeleteCustomerCommand(new CustomerId(customer.Id)));
            Customers.Remove(customer);
        }
    }

    [RelayCommand]
    private void NavigateToDetail(CustomerDto customer)
    {
        _navigation.NavigateTo<CustomerDetailPage>(customer.Id);
    }
}
```

---

## 7. マルチウィンドウ管理

```csharp
// WinUI 3 のマルチウィンドウ管理
using Microsoft.UI;
using Microsoft.UI.Windowing;
using WinRT.Interop;

public class WindowManager
{
    private readonly Dictionary<string, Window> _windows = new();

    /// <summary>
    /// 新しいウィンドウを作成・表示する
    /// </summary>
    public Window CreateWindow(string id, string title, Type pageType,
        int width = 800, int height = 600)
    {
        if (_windows.TryGetValue(id, out var existing))
        {
            // 既存ウィンドウをアクティブにする
            ActivateWindow(existing);
            return existing;
        }

        var window = new Window
        {
            Title = title,
            Content = (Page)App.GetService(pageType),
        };

        // AppWindow でサイズと位置を設定
        var appWindow = GetAppWindow(window);
        appWindow.Resize(new Windows.Graphics.SizeInt32(width, height));

        // ウィンドウが閉じられたときの処理
        window.Closed += (_, _) =>
        {
            _windows.Remove(id);
        };

        _windows[id] = window;
        window.Activate();
        return window;
    }

    /// <summary>
    /// IDを指定してウィンドウを取得
    /// </summary>
    public Window? GetWindow(string id) =>
        _windows.TryGetValue(id, out var w) ? w : null;

    /// <summary>
    /// 全ウィンドウを閉じる
    /// </summary>
    public void CloseAll()
    {
        foreach (var window in _windows.Values.ToList())
        {
            window.Close();
        }
        _windows.Clear();
    }

    private static AppWindow GetAppWindow(Window window)
    {
        var hwnd = WindowNative.GetWindowHandle(window);
        var windowId = Win32Interop.GetWindowIdFromWindow(hwnd);
        return AppWindow.GetFromWindowId(windowId);
    }

    private static void ActivateWindow(Window window)
    {
        var hwnd = WindowNative.GetWindowHandle(window);
        var windowId = Win32Interop.GetWindowIdFromWindow(hwnd);
        var appWindow = AppWindow.GetFromWindowId(windowId);
        // ウィンドウを前面に
        if (appWindow.Presenter is OverlappedPresenter presenter)
        {
            presenter.IsMinimizable = true;
            presenter.Restore();
        }
        window.Activate();
    }
}
```

```csharp
// Electron のマルチウィンドウ管理（TypeScript）
// ※ 比較のために .NET の後に掲載
```

```typescript
// main.ts — Electron マルチウィンドウ管理
class WindowManager {
  private windows = new Map<string, BrowserWindow>();

  createWindow(
    id: string,
    options: {
      title: string;
      url: string;
      width?: number;
      height?: number;
      parent?: BrowserWindow;
    }
  ): BrowserWindow {
    // 既存ウィンドウがあれば前面に
    const existing = this.windows.get(id);
    if (existing && !existing.isDestroyed()) {
      existing.focus();
      return existing;
    }

    const win = new BrowserWindow({
      width: options.width ?? 800,
      height: options.height ?? 600,
      title: options.title,
      parent: options.parent,
      webPreferences: {
        preload: path.join(__dirname, 'preload.js'),
        contextIsolation: true,
        nodeIntegration: false,
      },
    });

    win.loadURL(options.url);

    win.on('closed', () => {
      this.windows.delete(id);
    });

    this.windows.set(id, win);
    return win;
  }

  getWindow(id: string): BrowserWindow | undefined {
    const win = this.windows.get(id);
    return win && !win.isDestroyed() ? win : undefined;
  }

  closeAll(): void {
    for (const [id, win] of this.windows) {
      if (!win.isDestroyed()) win.close();
    }
    this.windows.clear();
  }

  // ウィンドウ間通信
  sendToWindow(id: string, channel: string, ...args: any[]): void {
    const win = this.getWindow(id);
    win?.webContents.send(channel, ...args);
  }

  // 全ウィンドウにブロードキャスト
  broadcast(channel: string, ...args: any[]): void {
    for (const [, win] of this.windows) {
      if (!win.isDestroyed()) {
        win.webContents.send(channel, ...args);
      }
    }
  }
}
```

---

## 8. プラグインアーキテクチャ

```csharp
// プラグインインターフェース
namespace MyApp.Plugins;

/// <summary>
/// プラグインの基本インターフェース
/// </summary>
public interface IPlugin
{
    string Id { get; }
    string Name { get; }
    string Version { get; }
    string Description { get; }

    /// <summary>
    /// プラグインを初期化する
    /// </summary>
    Task InitializeAsync(IPluginContext context);

    /// <summary>
    /// プラグインを破棄する
    /// </summary>
    Task ShutdownAsync();
}

/// <summary>
/// プラグインに提供するコンテキスト
/// </summary>
public interface IPluginContext
{
    /// <summary>
    /// メニューにアイテムを追加
    /// </summary>
    void RegisterMenuItem(string menuPath, string label, Action handler);

    /// <summary>
    /// コマンドパレットにコマンドを追加
    /// </summary>
    void RegisterCommand(string id, string label, Func<Task> handler);

    /// <summary>
    /// サイドバーにパネルを追加
    /// </summary>
    void RegisterSidebarPanel(string id, string title, Type panelType);

    /// <summary>
    /// イベントを購読
    /// </summary>
    IDisposable Subscribe<TEvent>(Action<TEvent> handler);

    /// <summary>
    /// 設定を読み書き
    /// </summary>
    IPluginSettings Settings { get; }
}

// プラグインローダー
public class PluginLoader
{
    private readonly List<IPlugin> _plugins = new();
    private readonly string _pluginDir;
    private readonly IPluginContext _context;

    public PluginLoader(string pluginDir, IPluginContext context)
    {
        _pluginDir = pluginDir;
        _context = context;
    }

    /// <summary>
    /// プラグインディレクトリからすべてのプラグインを読み込む
    /// </summary>
    public async Task LoadAllAsync()
    {
        if (!Directory.Exists(_pluginDir)) return;

        foreach (var dir in Directory.GetDirectories(_pluginDir))
        {
            var dllFiles = Directory.GetFiles(dir, "*.dll");
            foreach (var dll in dllFiles)
            {
                try
                {
                    await LoadPluginAsync(dll);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Failed to load plugin {dll}: {ex.Message}");
                }
            }
        }
    }

    private async Task LoadPluginAsync(string dllPath)
    {
        // AssemblyLoadContext で分離してロード
        var loadContext = new PluginLoadContext(dllPath);
        var assembly = loadContext.LoadFromAssemblyPath(dllPath);

        var pluginTypes = assembly.GetTypes()
            .Where(t => typeof(IPlugin).IsAssignableFrom(t) && !t.IsAbstract);

        foreach (var type in pluginTypes)
        {
            if (Activator.CreateInstance(type) is IPlugin plugin)
            {
                await plugin.InitializeAsync(_context);
                _plugins.Add(plugin);
            }
        }
    }

    /// <summary>
    /// 全プラグインをシャットダウン
    /// </summary>
    public async Task UnloadAllAsync()
    {
        foreach (var plugin in _plugins)
        {
            try
            {
                await plugin.ShutdownAsync();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error shutting down {plugin.Name}: {ex.Message}");
            }
        }
        _plugins.Clear();
    }
}

/// <summary>
/// プラグイン用の分離された AssemblyLoadContext
/// </summary>
public class PluginLoadContext : AssemblyLoadContext
{
    private readonly AssemblyDependencyResolver _resolver;

    public PluginLoadContext(string pluginPath) : base(isCollectible: true)
    {
        _resolver = new AssemblyDependencyResolver(pluginPath);
    }

    protected override Assembly? Load(AssemblyName assemblyName)
    {
        var assemblyPath = _resolver.ResolveAssemblyToPath(assemblyName);
        return assemblyPath is not null
            ? LoadFromAssemblyPath(assemblyPath)
            : null;
    }
}
```

---

## 9. Win32 メッセージループの理解

```
Win32 メッセージループ:

  ┌──────────┐     ┌───────────────┐     ┌──────────────┐
  │  OS      │────→│ メッセージキュー  │────→│ WndProc      │
  │ (入力)   │     │               │     │ (メッセージ処理)│
  │ マウス   │     │ WM_PAINT      │     │              │
  │ キーボード│     │ WM_KEYDOWN    │     │ switch(msg)  │
  │ タイマー │     │ WM_MOUSEMOVE  │     │  case ...    │
  └──────────┘     └───────────────┘     └──────────────┘

  GetMessage() → TranslateMessage() → DispatchMessage() → WndProc()
```

```csharp
// Win32 メッセージループの基本（P/Invoke）
// WPF/WinUI 3 では通常直接操作しないが、理解は重要

using System.Runtime.InteropServices;

public class Win32MessageLoop
{
    [StructLayout(LayoutKind.Sequential)]
    public struct MSG
    {
        public IntPtr hwnd;
        public uint message;
        public IntPtr wParam;
        public IntPtr lParam;
        public uint time;
        public POINT pt;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct POINT
    {
        public int x;
        public int y;
    }

    [DllImport("user32.dll")]
    private static extern bool GetMessage(
        out MSG lpMsg, IntPtr hWnd, uint wMsgFilterMin, uint wMsgFilterMax);

    [DllImport("user32.dll")]
    private static extern bool TranslateMessage(ref MSG lpMsg);

    [DllImport("user32.dll")]
    private static extern IntPtr DispatchMessage(ref MSG lpMsg);

    // 標準メッセージループ（参考用 — 実際にはフレームワークが管理）
    public static void RunMessageLoop()
    {
        MSG msg;
        while (GetMessage(out msg, IntPtr.Zero, 0, 0))
        {
            TranslateMessage(ref msg);
            DispatchMessage(ref msg);
        }
    }

    // よく使う Win32 メッセージ定数
    public const uint WM_PAINT = 0x000F;
    public const uint WM_CLOSE = 0x0010;
    public const uint WM_DESTROY = 0x0002;
    public const uint WM_KEYDOWN = 0x0100;
    public const uint WM_LBUTTONDOWN = 0x0201;
    public const uint WM_MOUSEMOVE = 0x0200;
    public const uint WM_SIZE = 0x0005;
    public const uint WM_COPYDATA = 0x004A;
    public const uint WM_USER = 0x0400;
    public const uint WM_APP = 0x8000;
}
```

```csharp
// WPF での Win32 メッセージフック（高度な使用例）
using System.Windows.Interop;

public partial class MainWindow : Window
{
    private HwndSource? _hwndSource;

    protected override void OnSourceInitialized(EventArgs e)
    {
        base.OnSourceInitialized(e);

        // ウィンドウハンドルを取得してメッセージフックを設定
        _hwndSource = PresentationSource.FromVisual(this) as HwndSource;
        _hwndSource?.AddHook(WndProc);
    }

    private IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam,
        IntPtr lParam, ref bool handled)
    {
        switch ((uint)msg)
        {
            case Win32MessageLoop.WM_COPYDATA:
                // 他のアプリケーションからのデータ受信
                HandleCopyData(lParam);
                handled = true;
                break;

            case Win32MessageLoop.WM_APP + 1:
                // カスタムメッセージの処理
                HandleCustomMessage(wParam, lParam);
                handled = true;
                break;
        }

        return IntPtr.Zero;
    }

    protected override void OnClosed(EventArgs e)
    {
        _hwndSource?.RemoveHook(WndProc);
        base.OnClosed(e);
    }
}
```

---

## 10. 状態管理パターン

```csharp
// アプリケーション状態管理 — Redux 風パターン
namespace MyApp.State;

// 状態の定義（Immutable）
public record AppState
{
    public IReadOnlyList<Customer> Customers { get; init; } = Array.Empty<Customer>();
    public Customer? SelectedCustomer { get; init; }
    public bool IsLoading { get; init; }
    public string? ErrorMessage { get; init; }
    public ThemeMode Theme { get; init; } = ThemeMode.System;
}

// アクションの定義
public abstract record AppAction;
public record LoadCustomersAction : AppAction;
public record CustomersLoadedAction(IReadOnlyList<Customer> Customers) : AppAction;
public record SelectCustomerAction(Customer Customer) : AppAction;
public record SetErrorAction(string Message) : AppAction;
public record ClearErrorAction : AppAction;
public record ChangeThemeAction(ThemeMode Theme) : AppAction;

// リデューサー
public static class AppReducer
{
    public static AppState Reduce(AppState state, AppAction action)
    {
        return action switch
        {
            LoadCustomersAction =>
                state with { IsLoading = true, ErrorMessage = null },

            CustomersLoadedAction a =>
                state with { Customers = a.Customers, IsLoading = false },

            SelectCustomerAction a =>
                state with { SelectedCustomer = a.Customer },

            SetErrorAction a =>
                state with { ErrorMessage = a.Message, IsLoading = false },

            ClearErrorAction =>
                state with { ErrorMessage = null },

            ChangeThemeAction a =>
                state with { Theme = a.Theme },

            _ => state,
        };
    }
}

// ストア
public class Store : ObservableObject
{
    private AppState _state = new();

    public AppState State
    {
        get => _state;
        private set => SetProperty(ref _state, value);
    }

    public void Dispatch(AppAction action)
    {
        State = AppReducer.Reduce(State, action);
    }

    // 非同期アクション（Thunk）
    public async Task DispatchAsync(
        Func<Func<AppAction, void>, Task> thunk)
    {
        await thunk(Dispatch);
    }
}

// 使用例
public partial class CustomerListViewModel : ObservableObject
{
    private readonly Store _store;
    private readonly ICustomerRepository _repository;

    public CustomerListViewModel(Store store, ICustomerRepository repository)
    {
        _store = store;
        _repository = repository;
    }

    [RelayCommand]
    private async Task LoadAsync()
    {
        _store.Dispatch(new LoadCustomersAction());
        try
        {
            var customers = await _repository.GetAllAsync();
            _store.Dispatch(new CustomersLoadedAction(customers));
        }
        catch (Exception ex)
        {
            _store.Dispatch(new SetErrorAction(ex.Message));
        }
    }
}
```

---

## 11. アンチパターン

```
よくある間違い:

  ✗ nodeIntegration: true にする
    → レンダラーから Node.js API に直接アクセス可能
    → XSS 攻撃で任意コード実行のリスク

  ✗ contextIsolation: false にする
    → preload のコンテキストが Web ページと共有
    → prototype pollution 攻撃のリスク

  ✗ ipcRenderer を丸ごと公開する
    → 任意のチャンネルにメッセージ送信可能
    → ホワイトリスト方式で制限すべき

  ✗ メインプロセスで入力検証しない
    → レンダラーからの入力は常に信頼できない
    → パストラバーサル、インジェクション攻撃のリスク

  ✗ UI スレッドで重い処理を実行する（.NET）
    → UIフリーズ（応答なしダイアログ）
    → Task.Run + async/await で回避

  ✗ ViewModel にフレームワーク依存コードを入れる
    → テスタビリティが低下
    → サービスインターフェースで抽象化すべき

  ✗ DI を使わずに new で依存性を生成する
    → 結合度が高く、モック化困難
    → コンストラクタインジェクションを使用する

  ✗ 状態を複数の場所に分散して管理する
    → 状態の不整合が発生しやすい
    → 単一のStore / ViewModel で一元管理する
```

---

## FAQ

### Q1: Electron と Tauri のセキュリティモデルの違いは？
Electron はデフォルトで緩い設定（手動で厳格化が必要）。Tauri はデフォルトで全て無効（必要なAPIのみ capabilities で許可）。Tauri の方がセキュア・バイ・デフォルト。

### Q2: preload スクリプトは複数使えるか？
Electron では BrowserWindow ごとに1つの preload を指定。複数の機能は1つの preload 内でモジュール化して管理する。

### Q3: IPC 通信のパフォーマンスは？
invoke/handle は数百μs程度のオーバーヘッド。大量データの転送は MessagePort や SharedArrayBuffer の活用を検討。

### Q4: WPF/WinUI 3 で DI コンテナは何を使うべきか？
Microsoft.Extensions.DependencyInjection が標準的。Generic Host パターンで構成する。Autofac や DryIoc も選択肢だが、特別な理由がなければ標準のもので十分。

### Q5: MVVM と MVC の違いは？
MVC はコントローラーが入力を受け取りモデルを操作する。MVVM は View と ViewModel がデータバインディングで結合され、ViewModel が表示ロジックを担当する。デスクトップの XAML アプリには MVVM が最適。

### Q6: CommunityToolkit.Mvvm と Prism の使い分けは？
CommunityToolkit.Mvvm は軽量で Source Generator ベース。Prism はモジュール化・リージョン管理・ダイアログサービスなど大規模向け機能が充実。小中規模は CommunityToolkit、大規模エンタープライズは Prism を検討。

### Q7: クリーンアーキテクチャはデスクトップアプリに必要か？
小規模アプリには過剰設計になりがち。中規模以上、または長期保守が見込まれるアプリには推奨。まずは MVVM + DI から始め、必要に応じてレイヤーを追加するのが現実的。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| プロセス分離 | メイン（バックエンド）とレンダラー（UI）を分離 |
| IPC | invoke/handle（Request-Response）が基本 |
| preload | contextBridge でホワイトリスト方式の API 公開 |
| セキュリティ | nodeIntegration:false + contextIsolation:true + sandbox:true |
| Tauri | Capabilities でAPI単位の権限管理 |
| .NET MVVM | CommunityToolkit.Mvvm + DI で疎結合な設計 |
| クリーンアーキテクチャ | ドメイン → アプリケーション → インフラ → プレゼンテーション |
| 状態管理 | Store パターンまたは ViewModel で一元管理 |
| マルチウィンドウ | WindowManager で ID ベースの管理 |
| プラグイン | AssemblyLoadContext で分離ロード |

---

## 次に読むべきガイド
→ [[02-native-features.md]] — ネイティブ機能の活用

---

## 参考文献
1. Electron. "Security." electronjs.org/docs/tutorial/security, 2024.
2. Electron. "Context Isolation." electronjs.org/docs/tutorial/context-isolation, 2024.
3. Tauri. "Security." tauri.app/security, 2024.
4. Microsoft. "Dependency Injection in .NET." learn.microsoft.com/dotnet/core/extensions/dependency-injection, 2024.
5. Microsoft. "Windows App SDK Architecture." learn.microsoft.com/windows/apps/windows-app-sdk, 2024.
6. Microsoft. "CommunityToolkit.Mvvm." learn.microsoft.com/dotnet/communitytoolkit/mvvm, 2024.
7. Jason Taylor. "Clean Architecture with .NET." github.com/jasontaylordev/CleanArchitecture, 2024.
8. Microsoft. "MSIX Packaging." learn.microsoft.com/windows/msix, 2024.

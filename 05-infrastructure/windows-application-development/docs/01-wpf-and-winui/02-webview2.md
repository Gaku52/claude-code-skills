# WebView2 統合

> Microsoft Edge (Chromium) ベースの WebView2 コントロールを使い、ネイティブアプリに Web コンテンツを埋め込むハイブリッドアプリケーション設計を学ぶ。

---

## この章で学ぶこと

1. **WebView2 コントロールの導入と基本設定**を理解し、WinUI 3 / WPF アプリに組み込めるようになる
2. **Web とネイティブ間の双方向通信**（JavaScript ↔ C#）を実装できるようになる
3. **セキュリティモデル**を理解し、安全なハイブリッドアプリを設計できるようになる
4. **パフォーマンス最適化**と**デバッグ手法**を習得し、本番品質のハイブリッドアプリを構築できるようになる

---

## 1. WebView2 とは何か

### 1.1 アーキテクチャ

```
+----------------------------------------------+
|            ホストアプリケーション               |
|  (WinUI 3 / WPF / WinForms / Win32)         |
|                                              |
|  +-----------------------------------------+ |
|  |           WebView2 コントロール           | |
|  |  +-----------------------------------+  | |
|  |  |   Chromium レンダリングエンジン     |  | |
|  |  |   (Edge WebView2 Runtime)         |  | |
|  |  +-----------------------------------+  | |
|  |       ↕ IPC (COM/JSON)                  | |
|  |  +-----------------------------------+  | |
|  |  |   ネイティブホスト (C# / C++)      |  | |
|  |  +-----------------------------------+  | |
|  +-----------------------------------------+ |
+----------------------------------------------+
```

WebView2 は Microsoft Edge と同じ Chromium エンジンを使用するが、アプリケーション内に**独立したブラウザプロセス**として動作する。これにより、最新の Web 標準をサポートしながらもネイティブアプリとの密な連携が可能になる。

### 1.2 プロセスモデルの詳細

WebView2 は複数のプロセスで構成されている。ホストアプリケーションは1つの Main プロセスで動作し、WebView2 はブラウザプロセス、GPU プロセス、ユーティリティプロセス、そしてレンダラープロセスを別途起動する。

```
+----------------------------------------------------------+
|  ホストアプリ (Main Process)                               |
|    └── CoreWebView2Environment                            |
|         └── CoreWebView2Controller                        |
|              └── CoreWebView2                             |
|                   ├── Browser Process (共有)              |
|                   │   ├── GPU Process                     |
|                   │   └── Utility Processes               |
|                   └── Renderer Process (WebView2 毎)      |
|                        └── V8 JavaScript Engine           |
+----------------------------------------------------------+
```

この分離モデルにより、Web コンテンツのクラッシュがホストアプリに影響を与えることなく、またセキュリティ境界が明確に維持される。

### 1.3 WebView2 と CEF / Electron の比較

| 項目 | WebView2 | CEF (CefSharp) | Electron |
|---|---|---|---|
| エンジン | Edge Chromium | Chromium (独自ビルド) | Chromium (同梱) |
| ランタイムサイズ | 共有 (0 MB追加*) | ~200 MB | ~150 MB |
| 更新方式 | OS/ランタイム更新 | アプリに同梱 | アプリに同梱 |
| ホスト言語 | C# / C++ / Win32 | C# | JavaScript/TypeScript |
| プロセスモデル | 分離プロセス | 分離プロセス | Main + Renderer |
| ライセンス | 無料 | BSD | MIT |
| 通信方式 | PostMessage / HostObject | CefSharp API | IPC (contextBridge) |
| マルチプラットフォーム | Windows のみ | Windows / macOS / Linux | Windows / macOS / Linux |
| メモリ使用量 | 低〜中（共有ランタイム） | 中〜高 | 高 |
| 起動速度 | 高速（ランタイム事前読込） | 中程度 | 低速 |

*Windows 11 には WebView2 Runtime がプリインストールされている。Windows 10 では別途インストールが必要。

---

## 2. セットアップ

### コード例 1: NuGet パッケージの追加

```xml
<!-- プロジェクトファイル (.csproj) に WebView2 パッケージを追加 -->
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0-windows10.0.19041.0</TargetFramework>
    <UseWinUI>true</UseWinUI>
  </PropertyGroup>

  <ItemGroup>
    <!-- WebView2 SDK -->
    <PackageReference Include="Microsoft.Web.WebView2" Version="1.0.2478.35" />
    <!-- Windows App SDK -->
    <PackageReference Include="Microsoft.WindowsAppSDK" Version="1.5.240627000" />
  </ItemGroup>
</Project>
```

### コード例 2: 基本的な WebView2 配置（WinUI 3）

```xml
<!-- WebView2Page.xaml — WebView2 を配置する XAML -->
<Page
    x:Class="HybridApp.WebView2Page"
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    xmlns:controls="using:Microsoft.UI.Xaml.Controls">

    <Grid>
        <Grid.RowDefinitions>
            <!-- ナビゲーションバー -->
            <RowDefinition Height="Auto" />
            <!-- WebView2 コンテンツ -->
            <RowDefinition Height="*" />
        </Grid.RowDefinitions>

        <!-- アドレスバー風の UI -->
        <StackPanel Grid.Row="0" Orientation="Horizontal"
                    Spacing="8" Padding="8">
            <Button Content="←" Click="GoBack_Click" />
            <Button Content="→" Click="GoForward_Click" />
            <Button Content="↻" Click="Reload_Click" />
            <TextBox x:Name="AddressBar" Width="400"
                     KeyDown="AddressBar_KeyDown" />
        </StackPanel>

        <!-- WebView2 コントロール -->
        <WebView2 x:Name="WebView" Grid.Row="1"
                  NavigationCompleted="WebView_NavigationCompleted" />
    </Grid>
</Page>
```

```csharp
// WebView2Page.xaml.cs — WebView2 の初期化と基本操作
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.Web.WebView2.Core;

namespace HybridApp;

public sealed partial class WebView2Page : Page
{
    public WebView2Page()
    {
        this.InitializeComponent();
        // ページ読み込み完了後に WebView2 を初期化
        this.Loaded += async (s, e) => await InitializeWebView();
    }

    private async Task InitializeWebView()
    {
        // WebView2 環境を初期化（ランタイムの検出と接続）
        await WebView.EnsureCoreWebView2Async();

        // 初期ページを表示
        WebView.CoreWebView2.Navigate("https://example.com");

        // 設定: 開発者ツールを有効化（デバッグ時のみ推奨）
        WebView.CoreWebView2.Settings.AreDevToolsEnabled = true;

        // 設定: コンテキストメニューを無効化（プロダクション向け）
        WebView.CoreWebView2.Settings.AreDefaultContextMenusEnabled = false;
    }

    // ナビゲーション完了時にアドレスバーを更新
    private void WebView_NavigationCompleted(
        WebView2 sender, CoreWebView2NavigationCompletedEventArgs args)
    {
        AddressBar.Text = WebView.CoreWebView2.Source;
    }

    private void GoBack_Click(object s, RoutedEventArgs e)
        => WebView.CoreWebView2?.GoBack();

    private void GoForward_Click(object s, RoutedEventArgs e)
        => WebView.CoreWebView2?.GoForward();

    private void Reload_Click(object s, RoutedEventArgs e)
        => WebView.CoreWebView2?.Reload();

    // Enter キーで URL に移動
    private void AddressBar_KeyDown(object s, KeyRoutedEventArgs e)
    {
        if (e.Key == Windows.System.VirtualKey.Enter)
        {
            WebView.CoreWebView2?.Navigate(AddressBar.Text);
        }
    }
}
```

### コード例 2b: WPF での WebView2 配置

```xml
<!-- MainWindow.xaml — WPF での WebView2 配置 -->
<Window x:Class="HybridApp.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        xmlns:wv2="clr-namespace:Microsoft.Web.WebView2.Wpf;assembly=Microsoft.Web.WebView2.Wpf"
        Title="WebView2 Hybrid App" Height="700" Width="1100">

    <Grid>
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="*"/>
            <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>

        <!-- ツールバー -->
        <ToolBar Grid.Row="0">
            <Button Content="戻る" Click="GoBack_Click"/>
            <Button Content="進む" Click="GoForward_Click"/>
            <Button Content="更新" Click="Reload_Click"/>
            <Separator/>
            <TextBox x:Name="UrlTextBox" Width="500"
                     KeyDown="UrlTextBox_KeyDown"/>
            <Button Content="移動" Click="Navigate_Click"/>
        </ToolBar>

        <!-- WebView2 コントロール (WPF 版) -->
        <wv2:WebView2 x:Name="WebView" Grid.Row="1"
                      Source="https://example.com"
                      NavigationCompleted="WebView_NavigationCompleted"
                      CoreWebView2InitializationCompleted="WebView_CoreWebView2InitializationCompleted"/>

        <!-- ステータスバー -->
        <StatusBar Grid.Row="2">
            <StatusBarItem>
                <TextBlock x:Name="StatusText" Text="準備完了"/>
            </StatusBarItem>
        </StatusBar>
    </Grid>
</Window>
```

```csharp
// MainWindow.xaml.cs — WPF 版の初期化コード
using System.Windows;
using System.Windows.Input;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.Wpf;

namespace HybridApp;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    // CoreWebView2 の初期化完了イベント
    private void WebView_CoreWebView2InitializationCompleted(
        object? sender, CoreWebView2InitializationCompletedEventArgs e)
    {
        if (!e.IsSuccess)
        {
            StatusText.Text = $"WebView2 初期化エラー: {e.InitializationException?.Message}";
            return;
        }

        var settings = WebView.CoreWebView2.Settings;
        settings.AreDevToolsEnabled = true;
        settings.IsStatusBarEnabled = false;
        settings.AreDefaultContextMenusEnabled = false;

        // ナビゲーション開始イベントの購読
        WebView.CoreWebView2.NavigationStarting += (s, args) =>
        {
            StatusText.Text = $"読み込み中: {args.Uri}";
        };

        // ダウンロード開始イベントの購読
        WebView.CoreWebView2.DownloadStarting += (s, args) =>
        {
            // ダウンロード先をカスタマイズ
            args.ResultFilePath = System.IO.Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                "HybridApp", "Downloads",
                System.IO.Path.GetFileName(args.ResultFilePath));
        };

        StatusText.Text = "WebView2 初期化完了";
    }

    private void WebView_NavigationCompleted(
        object? sender, CoreWebView2NavigationCompletedEventArgs e)
    {
        UrlTextBox.Text = WebView.CoreWebView2.Source;
        StatusText.Text = e.IsSuccess ? "読み込み完了" : $"エラー: {e.WebErrorStatus}";
    }

    private void GoBack_Click(object sender, RoutedEventArgs e)
        => WebView.CoreWebView2?.GoBack();

    private void GoForward_Click(object sender, RoutedEventArgs e)
        => WebView.CoreWebView2?.GoForward();

    private void Reload_Click(object sender, RoutedEventArgs e)
        => WebView.CoreWebView2?.Reload();

    private void Navigate_Click(object sender, RoutedEventArgs e)
        => NavigateToUrl();

    private void UrlTextBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter) NavigateToUrl();
    }

    private void NavigateToUrl()
    {
        var url = UrlTextBox.Text;
        if (!url.StartsWith("http://") && !url.StartsWith("https://"))
            url = "https://" + url;
        WebView.CoreWebView2?.Navigate(url);
    }
}
```

### コード例 2c: WebView2 環境のカスタム設定

```csharp
// カスタム環境設定での WebView2 初期化
private async Task InitializeWithCustomEnvironment()
{
    // ユーザーデータフォルダのカスタマイズ
    // （複数のWebView2 インスタンスを独立させる場合などに使用）
    var userDataFolder = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "HybridApp", "WebView2Data");

    // 環境オプションの設定
    var options = new CoreWebView2EnvironmentOptions
    {
        // 追加のブラウザ引数
        AdditionalBrowserArguments = "--disable-web-security=false --enable-features=msWebView2EnableDraggableRegions",
        // 言語設定
        Language = "ja",
        // プロキシ設定
        // AdditionalBrowserArguments = "--proxy-server=\"socks5://proxy.example.com:1080\"",
    };

    // カスタム環境で WebView2 を初期化
    var environment = await CoreWebView2Environment.CreateAsync(
        browserExecutableFolder: null,  // null = システムのEdgeを使用
        userDataFolder: userDataFolder,
        options: options);

    await WebView.EnsureCoreWebView2Async(environment);

    // クッキーマネージャーの設定
    var cookieManager = WebView.CoreWebView2.CookieManager;
    // 特定のクッキーを設定
    var cookie = cookieManager.CreateCookie(
        name: "session",
        value: "abc123",
        domain: "app.local",
        path: "/");
    cookie.IsSecure = true;
    cookie.IsHttpOnly = true;
    cookie.SameSite = CoreWebView2CookieSameSiteKind.Strict;
    cookieManager.AddOrUpdateCookie(cookie);
}
```

---

## 3. Web ↔ Native 通信

### 3.1 通信アーキテクチャ

```
+----------------------------+     +----------------------------+
|      Web (JavaScript)      |     |     Native (C#)            |
|                            |     |                            |
|  window.chrome.webview     |     |  CoreWebView2              |
|    .postMessage(json) ──────────→  .WebMessageReceived       |
|                            |     |                            |
|  window.chrome.webview     |     |  CoreWebView2              |
|    .addEventListener() ←────────  .PostWebMessageAsJson()    |
|                            |     |                            |
|  window.nativeApi          |     |  AddHostObjectToScript()   |
|    .methodCall() ────────────────→  [ComVisible] クラス       |
+----------------------------+     +----------------------------+
        ↕ PostMessage                     ↕ HostObject
   (非同期・JSON文字列)             (同期/非同期・直接呼出)
```

### コード例 3: PostMessage による双方向通信

```csharp
// Native 側: メッセージの送受信セットアップ
private async Task SetupMessaging()
{
    await WebView.EnsureCoreWebView2Async();

    // Web → Native: メッセージ受信ハンドラ
    WebView.CoreWebView2.WebMessageReceived += (sender, args) =>
    {
        // JSON 文字列としてメッセージを受信
        string message = args.WebMessageAsJson;
        var data = JsonSerializer.Deserialize<MessagePayload>(message);

        switch (data?.Type)
        {
            case "saveFile":
                // ネイティブのファイル保存ダイアログを使用
                HandleSaveFile(data.Content);
                break;
            case "getSystemInfo":
                // システム情報を Web 側に返送
                var info = new { os = Environment.OSVersion.ToString(),
                                 memory = GC.GetTotalMemory(false) };
                string response = JsonSerializer.Serialize(info);
                // Native → Web: メッセージ送信
                WebView.CoreWebView2.PostWebMessageAsJson(response);
                break;
        }
    };
}

// メッセージペイロードの型定義
record MessagePayload(string Type, string Content);
```

```javascript
// Web 側 (JavaScript): メッセージの送受信

// Native にメッセージを送信する関数
function sendToNative(type, content) {
  // chrome.webview.postMessage で Native 側にデータを送る
  window.chrome.webview.postMessage(
    JSON.stringify({ type, content })
  );
}

// Native からのメッセージを受信するリスナー
window.chrome.webview.addEventListener('message', (event) => {
  // event.data に Native から送られた JSON が入る
  const data = JSON.parse(event.data);
  console.log('ネイティブからの応答:', data);
  document.getElementById('result').textContent = JSON.stringify(data);
});

// 使用例: ファイル保存をリクエスト
document.getElementById('saveBtn').addEventListener('click', () => {
  sendToNative('saveFile', 'ここに保存する内容');
});

// 使用例: システム情報を取得
document.getElementById('infoBtn').addEventListener('click', () => {
  sendToNative('getSystemInfo', '');
});
```

### コード例 3b: 型安全な通信レイヤーの構築

```csharp
// Native 側: 構造化されたメッセージルーターの実装
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HybridApp.Communication;

// メッセージのベースクラス
public record BridgeMessage
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = Guid.NewGuid().ToString();

    [JsonPropertyName("type")]
    public string Type { get; init; } = "";

    [JsonPropertyName("payload")]
    public JsonElement? Payload { get; init; }
}

// レスポンスメッセージ
public record BridgeResponse
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = "";

    [JsonPropertyName("success")]
    public bool Success { get; init; }

    [JsonPropertyName("data")]
    public object? Data { get; init; }

    [JsonPropertyName("error")]
    public string? Error { get; init; }
}

// メッセージルーター: メッセージタイプに基づいてハンドラに振り分ける
public class MessageRouter
{
    private readonly Dictionary<string, Func<JsonElement?, Task<object?>>> _handlers = new();

    // ハンドラの登録
    public void Register(string messageType, Func<JsonElement?, Task<object?>> handler)
    {
        _handlers[messageType] = handler;
    }

    // 型安全なハンドラの登録
    public void Register<TPayload, TResult>(
        string messageType,
        Func<TPayload, Task<TResult>> handler) where TPayload : class
    {
        _handlers[messageType] = async (payload) =>
        {
            var typedPayload = payload.HasValue
                ? JsonSerializer.Deserialize<TPayload>(payload.Value.GetRawText())
                : null;

            if (typedPayload == null)
                throw new ArgumentException($"ペイロードのデシリアライズに失敗: {messageType}");

            return await handler(typedPayload);
        };
    }

    // メッセージのルーティング
    public async Task<BridgeResponse> Route(BridgeMessage message)
    {
        if (!_handlers.TryGetValue(message.Type, out var handler))
        {
            return new BridgeResponse
            {
                Id = message.Id,
                Success = false,
                Error = $"未知のメッセージタイプ: {message.Type}"
            };
        }

        try
        {
            var result = await handler(message.Payload);
            return new BridgeResponse
            {
                Id = message.Id,
                Success = true,
                Data = result
            };
        }
        catch (Exception ex)
        {
            return new BridgeResponse
            {
                Id = message.Id,
                Success = false,
                Error = ex.Message
            };
        }
    }
}

// 使用例: メッセージルーターのセットアップ
public class HybridBridge
{
    private readonly MessageRouter _router = new();
    private readonly CoreWebView2 _webView;

    public HybridBridge(CoreWebView2 webView)
    {
        _webView = webView;

        // ハンドラの登録
        _router.Register<SaveFileRequest, SaveFileResult>(
            "saveFile", HandleSaveFile);

        _router.Register<SearchRequest, SearchResult>(
            "search", HandleSearch);

        // WebView2 メッセージ受信の設定
        _webView.WebMessageReceived += OnWebMessageReceived;
    }

    private async void OnWebMessageReceived(object? sender,
        CoreWebView2WebMessageReceivedEventArgs args)
    {
        var message = JsonSerializer.Deserialize<BridgeMessage>(args.WebMessageAsJson);
        if (message == null) return;

        var response = await _router.Route(message);
        var responseJson = JsonSerializer.Serialize(response);
        _webView.PostWebMessageAsJson(responseJson);
    }

    private async Task<SaveFileResult> HandleSaveFile(SaveFileRequest request)
    {
        // ファイル保存の実装
        await File.WriteAllTextAsync(request.Path, request.Content);
        return new SaveFileResult { BytesWritten = request.Content.Length };
    }

    private async Task<SearchResult> HandleSearch(SearchRequest request)
    {
        // 検索の実装
        await Task.Delay(100); // シミュレーション
        return new SearchResult
        {
            Results = new[] { "結果1", "結果2", "結果3" },
            TotalCount = 3
        };
    }
}

// リクエスト・レスポンスの型定義
public record SaveFileRequest(string Path, string Content);
public record SaveFileResult { public int BytesWritten { get; init; } }
public record SearchRequest(string Query, int MaxResults);
public record SearchResult
{
    public string[] Results { get; init; } = Array.Empty<string>();
    public int TotalCount { get; init; }
}
```

```javascript
// Web 側: 型安全な通信ラッパー（TypeScript 形式で記述）

// ブリッジクラス: Promise ベースのリクエスト/レスポンス管理
class NativeBridge {
  constructor() {
    this.pendingRequests = new Map();

    // Native からのレスポンスを受信
    window.chrome.webview.addEventListener('message', (event) => {
      const response = JSON.parse(event.data);
      const pending = this.pendingRequests.get(response.id);
      if (pending) {
        this.pendingRequests.delete(response.id);
        if (response.success) {
          pending.resolve(response.data);
        } else {
          pending.reject(new Error(response.error));
        }
      }
    });
  }

  // Native にリクエストを送り、レスポンスを Promise で返す
  invoke(type, payload) {
    return new Promise((resolve, reject) => {
      const id = crypto.randomUUID();
      this.pendingRequests.set(id, { resolve, reject });

      window.chrome.webview.postMessage(JSON.stringify({
        id,
        type,
        payload
      }));

      // タイムアウト設定（30秒）
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error(`リクエストタイムアウト: ${type}`));
        }
      }, 30000);
    });
  }
}

// グローバルインスタンス
const bridge = new NativeBridge();

// 使用例
async function saveDocument(content) {
  try {
    const result = await bridge.invoke('saveFile', {
      path: 'document.txt',
      content: content
    });
    console.log(`保存完了: ${result.bytesWritten} バイト`);
  } catch (error) {
    console.error('保存エラー:', error.message);
  }
}

async function searchDocuments(query) {
  const result = await bridge.invoke('search', {
    query: query,
    maxResults: 10
  });
  return result.results;
}
```

### コード例 4: HostObject による直接メソッド呼び出し

```csharp
// Native 側: COM 可視のホストオブジェクトを定義
using System.Runtime.InteropServices;

namespace HybridApp;

// COM 経由で JavaScript から直接呼び出せるクラス
[ClassInterface(ClassInterfaceType.AutoDual)]
[ComVisible(true)]
public class NativeApi
{
    // ファイルの読み込み
    public string ReadFile(string path)
    {
        if (!IsPathAllowed(path))
            throw new UnauthorizedAccessException("許可されていないパスです");

        return File.ReadAllText(path);
    }

    // 通知の表示
    public void ShowNotification(string title, string message)
    {
        // Windows 通知 API を呼び出す
        new ToastContentBuilder()
            .AddText(title)
            .AddText(message)
            .Show();
    }

    // パスの許可チェック（セキュリティ）
    private bool IsPathAllowed(string path)
    {
        var allowedBase = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
            "HybridApp");
        return Path.GetFullPath(path).StartsWith(allowedBase);
    }
}

// WebView2 に HostObject を登録
private async Task RegisterHostObject()
{
    await WebView.EnsureCoreWebView2Async();

    // "nativeApi" という名前で JavaScript からアクセス可能にする
    WebView.CoreWebView2.AddHostObjectToScript("nativeApi", new NativeApi());
}
```

```javascript
// Web 側: HostObject を使ったネイティブ API 呼び出し

// HostObject へのプロキシを取得（非同期）
const nativeApi = window.chrome.webview.hostObjects.nativeApi;

// ネイティブメソッドを呼び出す（非同期で結果が返る）
async function readNativeFile() {
  try {
    const content = await nativeApi.ReadFile('C:\\Users\\docs\\data.txt');
    document.getElementById('fileContent').textContent = content;
  } catch (error) {
    console.error('ファイル読み込みエラー:', error);
  }
}

// 通知を表示（戻り値なし）
async function showNotification() {
  await nativeApi.ShowNotification(
    'タスク完了',
    'データの同期が完了しました'
  );
}
```

---

## 4. ハイブリッドアプリ設計パターン

### 4.1 設計パターンの比較

| パターン | 説明 | 適用場面 |
|---|---|---|
| Web Primary | UI の大部分を Web で構築し、ネイティブは薄いシェル | 既存 Web アプリのデスクトップ化 |
| Native Primary | ネイティブ UI が主体で、一部を WebView2 で表示 | ダッシュボード、レポート表示 |
| Hybrid Split | 画面ごとに Web / ネイティブを使い分け | 複雑なアプリで段階的移行中 |
| Micro Frontend | 複数の WebView2 で異なる Web アプリを統合 | マイクロサービス的 UI 統合 |

### コード例 5: ローカルコンテンツの配信

```csharp
// ローカルの HTML/JS/CSS をセキュアに配信する設定
private async Task SetupLocalContent()
{
    await WebView.EnsureCoreWebView2Async();

    // 仮想ホスト名とローカルフォルダのマッピングを設定
    // "app.local" というホスト名でアクセスするとローカルファイルが返される
    WebView.CoreWebView2.SetVirtualHostNameToFolderMapping(
        hostName: "app.local",
        folderPath: Path.Combine(AppContext.BaseDirectory, "WebContent"),
        accessKind: CoreWebView2HostResourceAccessKind.Allow
    );

    // ローカルコンテンツに仮想 URL でアクセス
    WebView.CoreWebView2.Navigate("https://app.local/index.html");
}
```

```
プロジェクトディレクトリ構成:

HybridApp/
├── HybridApp.csproj
├── App.xaml / App.xaml.cs
├── MainWindow.xaml / MainWindow.xaml.cs
├── NativeApi.cs                  ← HostObject 定義
├── WebContent/                   ← Web コンテンツ（ビルド成果物）
│   ├── index.html
│   ├── assets/
│   │   ├── app.js
│   │   └── style.css
│   └── images/
└── Services/
    ├── FileService.cs            ← ファイル操作サービス
    └── NotificationService.cs    ← 通知サービス
```

### コード例 5b: React SPA をローカルコンテンツとして統合

```csharp
// React ビルド成果物を WebView2 で表示する設定
private async Task SetupReactApp()
{
    await WebView.EnsureCoreWebView2Async();

    // React ビルド出力をマッピング
    var webContentPath = Path.Combine(AppContext.BaseDirectory, "wwwroot");

    WebView.CoreWebView2.SetVirtualHostNameToFolderMapping(
        hostName: "app.local",
        folderPath: webContentPath,
        accessKind: CoreWebView2HostResourceAccessKind.Allow
    );

    // API サーバーへのリクエストをインターセプト
    WebView.CoreWebView2.AddWebResourceRequestedFilter(
        "https://api.local/*",
        CoreWebView2WebResourceContext.All);

    WebView.CoreWebView2.WebResourceRequested += async (sender, args) =>
    {
        var deferral = args.GetDeferral();
        try
        {
            var uri = new Uri(args.Request.Uri);
            var apiPath = uri.PathAndQuery.TrimStart('/');

            // ローカルの API ハンドラにルーティング
            var (statusCode, responseBody) = await HandleApiRequest(
                args.Request.Method, apiPath, args.Request.Content);

            var stream = new MemoryStream(
                System.Text.Encoding.UTF8.GetBytes(responseBody));

            args.Response = WebView.CoreWebView2.Environment
                .CreateWebResourceResponse(
                    stream, statusCode, "OK", "Content-Type: application/json");
        }
        finally
        {
            deferral.Complete();
        }
    };

    WebView.CoreWebView2.Navigate("https://app.local/index.html");
}

// ローカル API リクエストの処理
private async Task<(int statusCode, string body)> HandleApiRequest(
    string method, string path, Stream? requestBody)
{
    // REST API スタイルのルーティング
    return path switch
    {
        "api/tasks" when method == "GET" =>
            (200, JsonSerializer.Serialize(await GetAllTasks())),
        "api/tasks" when method == "POST" =>
            (201, JsonSerializer.Serialize(await CreateTask(requestBody))),
        _ => (404, "{\"error\": \"Not Found\"}")
    };
}
```

### コード例 5c: 複数 WebView2 インスタンスの管理

```csharp
// 複数の WebView2 パネルを管理するマネージャー
public class WebViewPanelManager
{
    private readonly Dictionary<string, WebView2> _panels = new();
    private readonly CoreWebView2Environment _sharedEnvironment;

    public WebViewPanelManager(CoreWebView2Environment environment)
    {
        _sharedEnvironment = environment;
    }

    // 新しい WebView2 パネルを作成
    public async Task<WebView2> CreatePanel(
        string panelId, Panel container, string initialUrl)
    {
        if (_panels.ContainsKey(panelId))
        {
            return _panels[panelId];
        }

        var webView = new WebView2();
        container.Children.Add(webView);

        // 共有環境を使用して初期化（メモリ効率が良い）
        await webView.EnsureCoreWebView2Async(_sharedEnvironment);

        // パネル間通信用のスクリプトを注入
        await webView.CoreWebView2.AddScriptToExecuteOnDocumentCreatedAsync(@"
            window.panelId = '" + panelId + @"';
            window.sendToPanel = function(targetPanelId, message) {
                window.chrome.webview.postMessage(JSON.stringify({
                    type: 'panel-message',
                    source: window.panelId,
                    target: targetPanelId,
                    data: message
                }));
            };
        ");

        // パネル間メッセージのルーティング
        webView.CoreWebView2.WebMessageReceived += (sender, args) =>
        {
            var message = JsonSerializer.Deserialize<PanelMessage>(args.WebMessageAsJson);
            if (message?.Type == "panel-message" && _panels.TryGetValue(message.Target, out var targetPanel))
            {
                targetPanel.CoreWebView2.PostWebMessageAsJson(
                    JsonSerializer.Serialize(new { source = message.Source, data = message.Data }));
            }
        };

        webView.CoreWebView2.Navigate(initialUrl);
        _panels[panelId] = webView;

        return webView;
    }

    // パネルを破棄
    public void DestroyPanel(string panelId)
    {
        if (_panels.TryGetValue(panelId, out var webView))
        {
            webView.CoreWebView2?.Stop();
            webView.Dispose();
            _panels.Remove(panelId);
        }
    }

    // 全パネルにブロードキャスト
    public void Broadcast(string message)
    {
        foreach (var (_, panel) in _panels)
        {
            panel.CoreWebView2?.PostWebMessageAsJson(message);
        }
    }
}

record PanelMessage(string Type, string Source, string Target, JsonElement Data);
```

---

## 5. セキュリティ

### 5.1 セキュリティ設定の一覧

```
+------------------------------------------+
|         WebView2 セキュリティ層           |
+------------------------------------------+
|                                          |
|  1. ナビゲーション制御                    |
|     → 許可 URL のホワイトリスト           |
|                                          |
|  2. スクリプト実行制御                    |
|     → 信頼されたスクリプトのみ許可        |
|                                          |
|  3. HostObject アクセス制御              |
|     → 最小限の API のみ公開              |
|                                          |
|  4. コンテンツセキュリティポリシー        |
|     → CSP ヘッダーで XSS 防止           |
|                                          |
|  5. プロセス分離                          |
|     → WebView2 は別プロセスで実行        |
|                                          |
|  6. ダウンロード制御                      |
|     → 許可されたファイル種別のみ          |
|                                          |
|  7. 証明書検証                            |
|     → カスタム証明書の検証ロジック        |
+------------------------------------------+
```

### セキュリティ設定コード

```csharp
// WebView2 のセキュリティ設定を強化する
private async Task ConfigureSecurity()
{
    await WebView.EnsureCoreWebView2Async();
    var settings = WebView.CoreWebView2.Settings;

    // 本番環境では開発者ツールを無効化
    settings.AreDevToolsEnabled = false;

    // 右クリックメニューを無効化
    settings.AreDefaultContextMenusEnabled = false;

    // 組み込み PDF ビューアを無効化（不要な場合）
    settings.IsBuiltInErrorPageEnabled = false;

    // ステータスバーを無効化
    settings.IsStatusBarEnabled = false;

    // ナビゲーション制御: 許可ドメイン以外への遷移をブロック
    WebView.CoreWebView2.NavigationStarting += (sender, args) =>
    {
        var uri = new Uri(args.Uri);
        var allowedHosts = new[] { "app.local", "api.example.com" };

        if (!allowedHosts.Contains(uri.Host))
        {
            // 許可されていないドメインへの遷移をキャンセル
            args.Cancel = true;
            System.Diagnostics.Debug.WriteLine(
                $"ブロック: {args.Uri} は許可リストに含まれていません");
        }
    };

    // 新しいウィンドウのオープンをブロック
    WebView.CoreWebView2.NewWindowRequested += (sender, args) =>
    {
        // ポップアップを防止し、現在のウィンドウでナビゲーション
        args.Handled = true;
        WebView.CoreWebView2.Navigate(args.Uri);
    };
}
```

### コード例 5d: 高度なセキュリティ設定

```csharp
// コンテンツセキュリティポリシーの動的注入
private async Task SetupCSP()
{
    await WebView.EnsureCoreWebView2Async();

    // WebResourceResponseReceived でレスポンスヘッダーを監視
    WebView.CoreWebView2.WebResourceResponseReceived += (sender, args) =>
    {
        var headers = args.Response.Headers;
        // CSP ヘッダーの確認（デバッグ用）
        if (headers.Contains("Content-Security-Policy"))
        {
            var enumerator = headers.GetEnumerator();
            while (enumerator.MoveNext())
            {
                System.Diagnostics.Debug.WriteLine(
                    $"ヘッダー: {enumerator.Current.Key} = {enumerator.Current.Value}");
            }
        }
    };

    // ページ読み込み時に CSP を注入するスクリプト
    await WebView.CoreWebView2.AddScriptToExecuteOnDocumentCreatedAsync(@"
        // CSP のメタタグを追加
        const meta = document.createElement('meta');
        meta.httpEquiv = 'Content-Security-Policy';
        meta.content = ""default-src 'self' https://app.local; "" +
                       ""script-src 'self' https://app.local; "" +
                       ""style-src 'self' https://app.local 'unsafe-inline'; "" +
                       ""img-src 'self' https://app.local data:; "" +
                       ""connect-src 'self' https://api.example.com;"";
        document.head.prepend(meta);
    ");
}

// ダウンロード制御の実装
private void SetupDownloadControl()
{
    var allowedExtensions = new HashSet<string>
    {
        ".pdf", ".csv", ".xlsx", ".docx", ".txt", ".json"
    };

    WebView.CoreWebView2.DownloadStarting += (sender, args) =>
    {
        var ext = Path.GetExtension(args.ResultFilePath).ToLowerInvariant();

        if (!allowedExtensions.Contains(ext))
        {
            // 許可されていないファイル種別のダウンロードをキャンセル
            args.Cancel = true;
            System.Diagnostics.Debug.WriteLine(
                $"ダウンロードブロック: {args.ResultFilePath} (拡張子: {ext})");
            return;
        }

        // ダウンロード先をアプリのダウンロードフォルダに限定
        var safeDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
            "HybridApp", "Downloads");
        Directory.CreateDirectory(safeDir);
        args.ResultFilePath = Path.Combine(safeDir, Path.GetFileName(args.ResultFilePath));
    };
}

// 証明書エラーの処理
private void SetupCertificateHandling()
{
    WebView.CoreWebView2.ServerCertificateErrorDetected += (sender, args) =>
    {
        // 開発環境では自己署名証明書を許可（本番では禁止）
#if DEBUG
        if (args.RequestUri.StartsWith("https://localhost"))
        {
            args.Action = CoreWebView2ServerCertificateErrorAction.AlwaysAllow;
            return;
        }
#endif
        // 本番環境では証明書エラーを拒否
        args.Action = CoreWebView2ServerCertificateErrorAction.Cancel;
        System.Diagnostics.Debug.WriteLine(
            $"証明書エラー: {args.RequestUri} - {args.ErrorStatus}");
    };
}
```

---

## 6. パフォーマンス最適化

### 6.1 WebView2 の起動時間最適化

```csharp
// パフォーマンス最適化: 環境の事前作成
public class WebView2EnvironmentPool
{
    private static CoreWebView2Environment? _sharedEnvironment;
    private static readonly SemaphoreSlim _lock = new(1);

    // 環境をシングルトンとして事前作成
    public static async Task<CoreWebView2Environment> GetOrCreateAsync()
    {
        if (_sharedEnvironment != null) return _sharedEnvironment;

        await _lock.WaitAsync();
        try
        {
            if (_sharedEnvironment != null) return _sharedEnvironment;

            var userDataFolder = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "HybridApp", "WebView2");

            _sharedEnvironment = await CoreWebView2Environment.CreateAsync(
                browserExecutableFolder: null,
                userDataFolder: userDataFolder,
                options: new CoreWebView2EnvironmentOptions
                {
                    Language = "ja",
                });

            return _sharedEnvironment;
        }
        finally
        {
            _lock.Release();
        }
    }
}

// アプリ起動時に環境を事前ウォームアップ
public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        // バックグラウンドで WebView2 環境を事前に作成
        _ = WebView2EnvironmentPool.GetOrCreateAsync();
    }
}
```

### 6.2 JavaScript 実行の最適化

```csharp
// スクリプト実行の最適化テクニック
private async Task OptimizedScriptExecution()
{
    await WebView.EnsureCoreWebView2Async();

    // 最適化1: ページ作成時に一度だけスクリプトを注入
    // （毎回 ExecuteScriptAsync するより効率的）
    var scriptId = await WebView.CoreWebView2.AddScriptToExecuteOnDocumentCreatedAsync(@"
        // ヘルパー関数を事前定義
        window.__appBridge = {
            cache: new Map(),
            batchQueue: [],
            flushInterval: null,

            // バッチ処理: 複数のメッセージをまとめて送信
            queueMessage(msg) {
                this.batchQueue.push(msg);
                if (!this.flushInterval) {
                    this.flushInterval = setTimeout(() => {
                        this.flush();
                    }, 16); // 60fps に合わせたバッチ間隔
                }
            },

            flush() {
                if (this.batchQueue.length > 0) {
                    window.chrome.webview.postMessage(JSON.stringify({
                        type: 'batch',
                        messages: this.batchQueue
                    }));
                    this.batchQueue = [];
                }
                this.flushInterval = null;
            }
        };
    ");

    // 最適化2: DOM 要素の参照をキャッシュして繰り返しクエリを避ける
    await WebView.CoreWebView2.ExecuteScriptAsync(@"
        // よく使う DOM 参照をキャッシュ
        const elements = {
            output: document.getElementById('output'),
            status: document.getElementById('status'),
            list: document.getElementById('list')
        };
        window.__cachedElements = elements;
    ");
}

// バッチメッセージの受信処理
private void HandleBatchMessages(string jsonMessage)
{
    var batch = JsonSerializer.Deserialize<BatchMessage>(jsonMessage);
    if (batch?.Type == "batch" && batch.Messages != null)
    {
        foreach (var msg in batch.Messages)
        {
            ProcessSingleMessage(msg);
        }
    }
}

record BatchMessage(string Type, JsonElement[]? Messages);
```

### 6.3 メモリ管理

```csharp
// メモリ使用量の監視と管理
public class WebView2MemoryMonitor
{
    private readonly WebView2 _webView;
    private readonly Timer _monitorTimer;
    private const long MemoryWarningThreshold = 500 * 1024 * 1024; // 500MB

    public WebView2MemoryMonitor(WebView2 webView)
    {
        _webView = webView;
        _monitorTimer = new Timer(CheckMemory, null, TimeSpan.Zero, TimeSpan.FromMinutes(1));
    }

    private async void CheckMemory(object? state)
    {
        try
        {
            // JavaScript ヒープのメモリ使用量を取得
            var result = await _webView.CoreWebView2.ExecuteScriptAsync(@"
                JSON.stringify({
                    jsHeapSizeLimit: performance.memory?.jsHeapSizeLimit || 0,
                    totalJSHeapSize: performance.memory?.totalJSHeapSize || 0,
                    usedJSHeapSize: performance.memory?.usedJSHeapSize || 0
                })
            ");

            var memory = JsonSerializer.Deserialize<MemoryInfo>(result.Trim('"'));
            if (memory != null && memory.UsedJSHeapSize > MemoryWarningThreshold)
            {
                System.Diagnostics.Debug.WriteLine(
                    $"メモリ警告: JS ヒープ使用量 {memory.UsedJSHeapSize / 1024 / 1024}MB");

                // ガベージコレクションを促す
                await _webView.CoreWebView2.ExecuteScriptAsync("window.gc?.()");
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"メモリ監視エラー: {ex.Message}");
        }
    }

    public void Dispose()
    {
        _monitorTimer.Dispose();
    }
}

record MemoryInfo(long JsHeapSizeLimit, long TotalJSHeapSize, long UsedJSHeapSize);
```

---

## 7. デバッグとトラブルシューティング

### 7.1 デバッグツール

```csharp
// デバッグ用のヘルパー設定
private async Task SetupDebugging()
{
    await WebView.EnsureCoreWebView2Async();

#if DEBUG
    // DevTools を有効化
    WebView.CoreWebView2.Settings.AreDevToolsEnabled = true;

    // コンソールメッセージを C# 側でもキャプチャ
    WebView.CoreWebView2.ConsoleMessageReceived += (sender, args) =>
    {
        var level = args.Level switch
        {
            CoreWebView2ConsoleMessageLevel.Log => "LOG",
            CoreWebView2ConsoleMessageLevel.Warning => "WARN",
            CoreWebView2ConsoleMessageLevel.Error => "ERROR",
            CoreWebView2ConsoleMessageLevel.Info => "INFO",
            _ => "DEBUG"
        };
        System.Diagnostics.Debug.WriteLine(
            $"[WebView2 {level}] {args.Message} ({args.Source}:{args.LineNumber})");
    };

    // プロセスエラーの監視
    WebView.CoreWebView2.ProcessFailed += (sender, args) =>
    {
        System.Diagnostics.Debug.WriteLine(
            $"WebView2 プロセスエラー: {args.ProcessFailedKind} - {args.Reason}");

        switch (args.ProcessFailedKind)
        {
            case CoreWebView2ProcessFailedKind.BrowserProcessExited:
                // ブラウザプロセスが異常終了した場合はアプリを再起動
                System.Diagnostics.Debug.WriteLine("ブラウザプロセスが終了しました。再起動が必要です。");
                break;
            case CoreWebView2ProcessFailedKind.RenderProcessExited:
            case CoreWebView2ProcessFailedKind.RenderProcessUnresponsive:
                // レンダラープロセスが異常終了した場合はリロード
                WebView.CoreWebView2.Reload();
                break;
        }
    };

    // パフォーマンスログの有効化
    WebView.CoreWebView2.NavigationStarting += (s, args) =>
    {
        _navigationStartTime = DateTime.UtcNow;
    };

    WebView.CoreWebView2.NavigationCompleted += (s, args) =>
    {
        var duration = DateTime.UtcNow - _navigationStartTime;
        System.Diagnostics.Debug.WriteLine(
            $"ナビゲーション完了: {duration.TotalMilliseconds}ms (成功: {args.IsSuccess})");
    };
#endif
}

private DateTime _navigationStartTime;
```

### 7.2 一般的な問題と解決策

```
+----------------------------------------------------------+
| 問題                        | 原因             | 解決策    |
+----------------------------------------------------------+
| WebView2 が表示されない      | ランタイム未インストール | ランタイム検出 + フォールバック |
| PostMessage が届かない       | タイミング問題    | NavigationCompleted 後に送信 |
| HostObject メソッドがない     | COM 登録漏れ     | ComVisible + ClassInterface |
| JavaScript エラーが見えない   | コンソール未購読  | ConsoleMessageReceived |
| メモリリーク                  | イベント未解除    | Dispose で明示解除 |
| 画面がちらつく               | 初期化の表示タイミング | show: false + ready-to-show |
| ナビゲーションが遅い         | キャッシュ無効    | CacheControl 設定 |
+----------------------------------------------------------+
```

---

## 8. WebView2 ランタイムの配布戦略

### 8.1 配布モデルの比較

| モデル | 説明 | アプリサイズ | 更新 |
|---|---|---|---|
| Evergreen (推奨) | OS のランタイムを使用 | +0 MB | 自動 |
| Fixed Version | 特定バージョンを同梱 | +150 MB | 手動 |
| Bootstrapper | 初回起動時にダウンロード | +2 MB | 自動 |

### コード例 8a: ランタイム検出とブートストラッパー

```csharp
// WebView2 ランタイムの検出と自動インストール
public static class WebView2RuntimeChecker
{
    public static bool IsRuntimeInstalled()
    {
        try
        {
            var version = CoreWebView2Environment.GetAvailableBrowserVersionString();
            return !string.IsNullOrEmpty(version);
        }
        catch
        {
            return false;
        }
    }

    public static async Task EnsureRuntimeAsync()
    {
        if (IsRuntimeInstalled()) return;

        var result = MessageBox.Show(
            "このアプリケーションには WebView2 ランタイムが必要です。\n" +
            "今すぐダウンロードしてインストールしますか？",
            "WebView2 ランタイムが必要です",
            MessageBoxButton.YesNo,
            MessageBoxImage.Question);

        if (result == MessageBoxResult.Yes)
        {
            await DownloadAndInstallRuntime();
        }
        else
        {
            Application.Current.Shutdown();
        }
    }

    private static async Task DownloadAndInstallRuntime()
    {
        var bootstrapperUrl = "https://go.microsoft.com/fwlink/p/?LinkId=2124703";
        var tempPath = Path.Combine(Path.GetTempPath(), "MicrosoftEdgeWebview2Setup.exe");

        using var httpClient = new HttpClient();
        var data = await httpClient.GetByteArrayAsync(bootstrapperUrl);
        await File.WriteAllBytesAsync(tempPath, data);

        var process = Process.Start(new ProcessStartInfo
        {
            FileName = tempPath,
            Arguments = "/silent /install",
            UseShellExecute = true,
            Verb = "runas" // 管理者権限で実行
        });

        await process!.WaitForExitAsync();

        if (!IsRuntimeInstalled())
        {
            MessageBox.Show(
                "WebView2 ランタイムのインストールに失敗しました。\n" +
                "手動でインストールしてください。",
                "エラー",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
        }
    }
}
```

---

## 9. アンチパターン

### アンチパターン 1: HostObject で全 API を無制限に公開する

```csharp
// NG: ファイルシステム全体にアクセス可能な API を公開
[ComVisible(true)]
public class UnsafeApi
{
    // 任意のパスのファイルを読み書きできてしまう
    public string ReadAnyFile(string path) => File.ReadAllText(path);
    public void WriteAnyFile(string path, string content) => File.WriteAllText(path, content);
    // レジストリや環境変数まで公開 → セキュリティリスク大
    public string GetEnvVar(string name) => Environment.GetEnvironmentVariable(name) ?? "";
}
```

```csharp
// OK: 最小権限原則に基づき、必要な API のみを安全に公開
[ComVisible(true)]
public class SafeApi
{
    private readonly string _sandboxDir;

    public SafeApi(string sandboxDir)
    {
        _sandboxDir = sandboxDir;
    }

    // サンドボックスディレクトリ内のみ読み取り可能
    public string ReadFile(string relativePath)
    {
        var fullPath = Path.GetFullPath(
            Path.Combine(_sandboxDir, relativePath));

        // パストラバーサル攻撃を防止
        if (!fullPath.StartsWith(_sandboxDir))
            throw new UnauthorizedAccessException("サンドボックス外のアクセスは禁止");

        return File.ReadAllText(fullPath);
    }
}
```

### アンチパターン 2: WebView2 Runtime の存在を確認しない

```csharp
// NG: ランタイムがインストールされていない環境でクラッシュ
public MainWindow()
{
    InitializeComponent();
    WebView.Source = new Uri("https://example.com"); // ← ランタイム未検出でクラッシュ
}
```

```csharp
// OK: ランタイムの存在をチェックしてフォールバック
public MainWindow()
{
    InitializeComponent();
    CheckWebView2Runtime();
}

private async void CheckWebView2Runtime()
{
    try
    {
        // ランタイムのバージョンを取得して存在を確認
        var version = CoreWebView2Environment.GetAvailableBrowserVersionString();
        await WebView.EnsureCoreWebView2Async();
        WebView.CoreWebView2.Navigate("https://example.com");
    }
    catch (WebView2RuntimeNotFoundException)
    {
        // ランタイム未インストール時のフォールバック
        ShowFallbackUI("WebView2 ランタイムがインストールされていません。"
            + "ダウンロードページを開きますか？");
    }
}
```

### アンチパターン 3: ナビゲーション完了前に通信を開始する

```csharp
// NG: WebView2 の初期化完了前にメッセージを送信
public MainWindow()
{
    InitializeComponent();
    // この時点ではまだ CoreWebView2 が null
    WebView.CoreWebView2.PostWebMessageAsJson("{}"); // NullReferenceException
}
```

```csharp
// OK: 初期化完了を待ってから通信を開始
public MainWindow()
{
    InitializeComponent();
    Loaded += async (s, e) =>
    {
        await WebView.EnsureCoreWebView2Async();
        WebView.CoreWebView2.NavigationCompleted += (sender, args) =>
        {
            if (args.IsSuccess)
            {
                // ナビゲーション完了後にメッセージを送信
                WebView.CoreWebView2.PostWebMessageAsJson(
                    JsonSerializer.Serialize(new { type = "init", version = "1.0" }));
            }
        };
        WebView.CoreWebView2.Navigate("https://app.local/index.html");
    };
}
```

### アンチパターン 4: イベントハンドラを解除しない

```csharp
// NG: ページ遷移のたびにイベントハンドラが蓄積する
private void SetupPage()
{
    // この関数が呼ばれるたびにハンドラが追加される → メモリリーク
    WebView.CoreWebView2.WebMessageReceived += OnMessageReceived;
    WebView.CoreWebView2.NavigationCompleted += OnNavigationCompleted;
}
```

```csharp
// OK: イベントハンドラを適切に管理する
private EventHandler<CoreWebView2WebMessageReceivedEventArgs>? _messageHandler;
private EventHandler<CoreWebView2NavigationCompletedEventArgs>? _navigationHandler;

private void SetupPage()
{
    // 既存のハンドラを解除してから登録
    CleanupHandlers();

    _messageHandler = OnMessageReceived;
    _navigationHandler = OnNavigationCompleted;

    WebView.CoreWebView2.WebMessageReceived += _messageHandler;
    WebView.CoreWebView2.NavigationCompleted += _navigationHandler;
}

private void CleanupHandlers()
{
    if (_messageHandler != null)
        WebView.CoreWebView2.WebMessageReceived -= _messageHandler;
    if (_navigationHandler != null)
        WebView.CoreWebView2.NavigationCompleted -= _navigationHandler;
}

// ウィンドウ破棄時にクリーンアップ
protected override void OnClosed(EventArgs e)
{
    CleanupHandlers();
    WebView?.Dispose();
    base.OnClosed(e);
}
```

---

## 10. FAQ

### Q1: WebView2 Runtime はアプリに同梱できるか？

**A:** はい。「固定バージョン配布」モードを使うと、特定バージョンの WebView2 Runtime をアプリに同梱できる。ただしサイズが約 150MB 増加し、セキュリティ更新を自分で管理する必要がある。通常は「Evergreen 配布」（OS 側で自動更新）が推奨される。

### Q2: WebView2 でローカルファイルに直接アクセスできるか？

**A:** `file://` プロトコルは既定で制限されている。ローカルコンテンツを配信するには `SetVirtualHostNameToFolderMapping()` を使って仮想ホスト名を割り当てるのが安全かつ推奨される方法である。これにより CORS 問題も回避できる。

### Q3: WebView2 のパフォーマンスは Electron と比べてどうか？

**A:** WebView2 は OS にインストール済みの共有ランタイムを使うため、アプリサイズが大幅に小さくなる。メモリ使用量も Electron より少ない傾向にある（Electron は各アプリに Chromium を同梱するため）。レンダリング性能自体は同じ Chromium エンジンのためほぼ同等である。

### Q4: WebView2 で React / Vue / Angular などのフレームワークを使えるか？

**A:** はい。WebView2 は Chromium ベースのため、React、Vue、Angular、Svelte など任意の Web フレームワークが問題なく動作する。開発時は Vite や Webpack の DevServer で Hot Module Replacement を使い、本番では `SetVirtualHostNameToFolderMapping` でビルド成果物を配信するのが一般的なワークフローである。

### Q5: 複数の WebView2 インスタンスを同時に使えるか？

**A:** はい。同一のアプリケーション内で複数の WebView2 コントロールを配置できる。`CoreWebView2Environment` を共有することで、ブラウザプロセスを共有しメモリ効率が向上する。ただし、各 WebView2 は独立したレンダラープロセスを持つため、インスタンス数が増えるとメモリ使用量も増加する点に注意が必要である。

### Q6: WebView2 で印刷機能を実装するにはどうすべきか？

**A:** `CoreWebView2.PrintAsync()` メソッドを使用する。`CoreWebView2PrintSettings` でページサイズ、余白、向き、ヘッダー/フッターなどをカスタマイズできる。`PrintToPdfAsync()` を使えば PDF としてエクスポートすることも可能である。

```csharp
// 印刷の実装例
private async Task PrintDocument()
{
    var printSettings = WebView.CoreWebView2.Environment.CreatePrintSettings();
    printSettings.Orientation = CoreWebView2PrintOrientation.Portrait;
    printSettings.ScaleFactor = 1.0;
    printSettings.ShouldPrintBackgrounds = true;
    printSettings.ShouldPrintHeaderAndFooter = false;

    // プリンターダイアログを表示して印刷
    var result = await WebView.CoreWebView2.PrintAsync(printSettings);

    // または PDF に出力
    await WebView.CoreWebView2.PrintToPdfAsync(
        Path.Combine(Environment.GetFolderPath(
            Environment.SpecialFolder.MyDocuments), "output.pdf"),
        printSettings);
}
```

---

## 11. まとめ

| トピック | キーポイント |
|---|---|
| WebView2 の役割 | Edge Chromium エンジンをアプリに埋め込む公式コントロール |
| セットアップ | NuGet パッケージ追加 + EnsureCoreWebView2Async() |
| PostMessage 通信 | JSON 文字列の非同期メッセージング。疎結合で推奨 |
| HostObject | COM 経由の直接メソッド呼び出し。高機能だが要セキュリティ対策 |
| ローカルコンテンツ | SetVirtualHostNameToFolderMapping で安全に配信 |
| セキュリティ | ナビゲーション制限 + API 最小公開 + CSP が三本柱 |
| ランタイム | Evergreen（自動更新）推奨。固定バージョンも選択可 |
| パフォーマンス | 環境共有、バッチメッセージング、スクリプト事前注入 |
| デバッグ | ConsoleMessageReceived + ProcessFailed でエラー検出 |
| WPF 対応 | Microsoft.Web.WebView2.Wpf パッケージで WPF アプリにも統合可 |

---

## 次に読むべきガイド

- **[00-electron-setup.md](../02-electron-and-tauri/00-electron-setup.md)** — Web 技術でデスクトップアプリを構築する Electron の入門
- **[02-tauri-setup.md](../02-electron-and-tauri/02-tauri-setup.md)** — 軽量な Rust ベースの代替フレームワーク Tauri

---

## 参考文献

1. Microsoft, "WebView2 — Introduction", https://learn.microsoft.com/microsoft-edge/webview2/
2. Microsoft, "WebView2 API Reference", https://learn.microsoft.com/microsoft-edge/webview2/reference/winrt/microsoft_web_webview2_core/
3. Microsoft, "WebView2 Sample Apps", https://github.com/AltF5/AltF5-WebView2-Sample
4. Microsoft, "Web/Native Interop", https://learn.microsoft.com/microsoft-edge/webview2/how-to/communicate-btwn-web-native
5. Microsoft, "WebView2 Distribution", https://learn.microsoft.com/microsoft-edge/webview2/concepts/distribution
6. Microsoft, "WebView2 Security Best Practices", https://learn.microsoft.com/microsoft-edge/webview2/concepts/security

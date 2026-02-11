# WebView2 統合

> Microsoft Edge (Chromium) ベースの WebView2 コントロールを使い、ネイティブアプリに Web コンテンツを埋め込むハイブリッドアプリケーション設計を学ぶ。

---

## この章で学ぶこと

1. **WebView2 コントロールの導入と基本設定**を理解し、WinUI 3 / WPF アプリに組み込めるようになる
2. **Web とネイティブ間の双方向通信**（JavaScript ↔ C#）を実装できるようになる
3. **セキュリティモデル**を理解し、安全なハイブリッドアプリを設計できるようになる

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

### 1.2 WebView2 と CEF / Electron の比較

| 項目 | WebView2 | CEF (CefSharp) | Electron |
|---|---|---|---|
| エンジン | Edge Chromium | Chromium (独自ビルド) | Chromium (同梱) |
| ランタイムサイズ | 共有 (0 MB追加*) | ~200 MB | ~150 MB |
| 更新方式 | OS/ランタイム更新 | アプリに同梱 | アプリに同梱 |
| ホスト言語 | C# / C++ / Win32 | C# | JavaScript/TypeScript |
| プロセスモデル | 分離プロセス | 分離プロセス | Main + Renderer |
| ライセンス | 無料 | BSD | MIT |
| 通信方式 | PostMessage / HostObject | CefSharp API | IPC (contextBridge) |

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

---

## 6. アンチパターン

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

---

## 7. FAQ

### Q1: WebView2 Runtime はアプリに同梱できるか？

**A:** はい。「固定バージョン配布」モードを使うと、特定バージョンの WebView2 Runtime をアプリに同梱できる。ただしサイズが約 150MB 増加し、セキュリティ更新を自分で管理する必要がある。通常は「Evergreen 配布」（OS 側で自動更新）が推奨される。

### Q2: WebView2 でローカルファイルに直接アクセスできるか？

**A:** `file://` プロトコルは既定で制限されている。ローカルコンテンツを配信するには `SetVirtualHostNameToFolderMapping()` を使って仮想ホスト名を割り当てるのが安全かつ推奨される方法である。これにより CORS 問題も回避できる。

### Q3: WebView2 のパフォーマンスは Electron と比べてどうか？

**A:** WebView2 は OS にインストール済みの共有ランタイムを使うため、アプリサイズが大幅に小さくなる。メモリ使用量も Electron より少ない傾向にある（Electron は各アプリに Chromium を同梱するため）。レンダリング性能自体は同じ Chromium エンジンのためほぼ同等である。

---

## 8. まとめ

| トピック | キーポイント |
|---|---|
| WebView2 の役割 | Edge Chromium エンジンをアプリに埋め込む公式コントロール |
| セットアップ | NuGet パッケージ追加 + EnsureCoreWebView2Async() |
| PostMessage 通信 | JSON 文字列の非同期メッセージング。疎結合で推奨 |
| HostObject | COM 経由の直接メソッド呼び出し。高機能だが要セキュリティ対策 |
| ローカルコンテンツ | SetVirtualHostNameToFolderMapping で安全に配信 |
| セキュリティ | ナビゲーション制限 + API 最小公開 + CSP が三本柱 |
| ランタイム | Evergreen（自動更新）推奨。固定バージョンも選択可 |

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

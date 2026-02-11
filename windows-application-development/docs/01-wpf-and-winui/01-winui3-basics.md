# WinUI 3 の基本

> Windows App SDK に含まれる最新の UI フレームワーク WinUI 3 を使い、Fluent Design に準拠したデスクトップアプリケーションを構築する方法を体系的に学ぶ。

---

## この章で学ぶこと

1. **WinUI 3 プロジェクトの作成**からビルド・実行までの一連のワークフローを理解する
2. **XAML の基礎構文**とデータバインディング、主要コントロールの使い方を習得する
3. **Fluent Design System** のスタイル・テーマ・ナビゲーションパターンを実装できるようになる

---

## 1. WinUI 3 とは何か

### 1.1 位置づけ

```
+--------------------------------------------------+
|              Windows App SDK                     |
|  +--------------------------------------------+  |
|  |              WinUI 3                        |  |
|  |  +----------+  +----------+  +-----------+ |  |
|  |  |  XAML    |  | Controls |  |  Fluent   | |  |
|  |  |  Engine  |  |  Library |  |  Design   | |  |
|  |  +----------+  +----------+  +-----------+ |  |
|  +--------------------------------------------+  |
|  +------------+  +-------------+  +-----------+  |
|  | App        |  | Windowing   |  | MRT       |  |
|  | Lifecycle  |  | (AppWindow) |  | (リソース)|  |
|  +------------+  +-------------+  +-----------+  |
+--------------------------------------------------+
```

WinUI 3 は **Windows App SDK** の一部であり、UWP の XAML 技術を Win32 デスクトップアプリから利用可能にしたものである。WPF とは異なるレンダリングエンジンを持ち、DirectX ベースの高速描画を実現する。

### 1.2 WPF / UWP / WinUI 3 の比較

| 項目 | WPF | UWP | WinUI 3 |
|---|---|---|---|
| 対象フレームワーク | .NET Framework / .NET | UWP (.NET Native) | .NET 6+ |
| XAML バージョン | WPF XAML | UWP XAML | WinUI XAML |
| 配布方式 | exe / MSI / MSIX | MSIX のみ | exe / MSIX |
| サンドボックス | なし | あり（厳格） | なし（任意で MSIX） |
| 最新 UI コントロール | 手動追加が必要 | 一部のみ | 完全サポート |
| Win32 API 呼び出し | 自由 | 制限あり | 自由 |
| 推奨用途 | レガシー保守 | ストアアプリ | 新規開発全般 |

---

## 2. プロジェクトの作成

### 2.1 前提条件

- Visual Studio 2022 17.8 以降
- Windows App SDK 拡張機能（NuGet: `Microsoft.WindowsAppSDK`）
- .NET 8 SDK 以降

### 2.2 テンプレートからの作成

```
Visual Studio → 新しいプロジェクトの作成
  → "Blank App, Packaged (WinUI 3 in Desktop)" を選択
  → プロジェクト名: MyFirstWinUI
  → ターゲットフレームワーク: net8.0-windows10.0.19041.0
```

### コード例 1: App.xaml.cs ― アプリケーションエントリポイント

```csharp
// App.xaml.cs — アプリケーションのエントリポイント
using Microsoft.UI.Xaml;

namespace MyFirstWinUI;

public partial class App : Application
{
    private Window? _window;

    public App()
    {
        // XAML コンポーネントを初期化
        this.InitializeComponent();
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        // メインウィンドウを生成して表示
        _window = new MainWindow();
        _window.Activate();
    }
}
```

### コード例 2: MainWindow.xaml ― 最初の XAML ページ

```xml
<!-- MainWindow.xaml — メインウィンドウの XAML 定義 -->
<Window
    x:Class="MyFirstWinUI.MainWindow"
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    Title="はじめての WinUI 3">

    <!-- StackPanel で垂直にコントロールを配置 -->
    <StackPanel
        HorizontalAlignment="Center"
        VerticalAlignment="Center"
        Spacing="16">

        <!-- テキスト表示 -->
        <TextBlock
            Text="Hello, WinUI 3!"
            Style="{StaticResource TitleTextBlockStyle}" />

        <!-- ボタン：クリックイベントをコードビハインドで処理 -->
        <Button
            x:Name="ClickMeButton"
            Content="クリックしてください"
            Click="ClickMeButton_Click" />

        <!-- 結果表示用テキスト -->
        <TextBlock x:Name="ResultText" />
    </StackPanel>
</Window>
```

```csharp
// MainWindow.xaml.cs — コードビハインド
using Microsoft.UI.Xaml;

namespace MyFirstWinUI;

public sealed partial class MainWindow : Window
{
    private int _clickCount = 0;

    public MainWindow()
    {
        this.InitializeComponent();
    }

    // ボタンクリック時のイベントハンドラ
    private void ClickMeButton_Click(object sender, RoutedEventArgs e)
    {
        _clickCount++;
        ResultText.Text = $"クリック回数: {_clickCount}";
    }
}
```

---

## 3. XAML の基礎

### 3.1 XAML の構造

```
<Window>                          ← ルート要素
  ├─ <StackPanel>                 ← レイアウトパネル
  │   ├─ <TextBlock Text="..." />  ← コンテンツ要素
  │   ├─ <Button Content="..." />  ← インタラクティブ要素
  │   └─ <Image Source="..." />    ← メディア要素
  └─ <Window.Resources>           ← リソース定義
      └─ <Style TargetType="..." /> ← スタイル
```

### 3.2 レイアウトパネルの比較

| パネル | 配置方式 | 主な用途 |
|---|---|---|
| `StackPanel` | 水平 or 垂直に直列配置 | 単純なフォーム、ツールバー |
| `Grid` | 行と列のセル配置 | 複雑なレイアウト全般 |
| `RelativePanel` | 相対位置指定 | レスポンシブ配置 |
| `Canvas` | 絶対座標指定 | 描画系、ドラッグ操作 |
| `WrapPanel`* | 折り返し配置 | タグ一覧、サムネイル |

*WinUI 3 Community Toolkit で提供

### コード例 3: Grid レイアウト

```xml
<!-- Grid を使った 2 列レイアウト -->
<Grid ColumnSpacing="16" RowSpacing="8" Padding="24">
    <Grid.ColumnDefinitions>
        <!-- 左列: ラベル（幅自動） -->
        <ColumnDefinition Width="Auto" />
        <!-- 右列: 入力（残り全て） -->
        <ColumnDefinition Width="*" />
    </Grid.ColumnDefinitions>
    <Grid.RowDefinitions>
        <RowDefinition Height="Auto" />
        <RowDefinition Height="Auto" />
        <RowDefinition Height="Auto" />
    </Grid.RowDefinitions>

    <!-- 行0: 名前入力 -->
    <TextBlock Grid.Row="0" Grid.Column="0"
               Text="名前:" VerticalAlignment="Center" />
    <TextBox Grid.Row="0" Grid.Column="1"
             PlaceholderText="山田太郎" />

    <!-- 行1: メールアドレス入力 -->
    <TextBlock Grid.Row="1" Grid.Column="0"
               Text="メール:" VerticalAlignment="Center" />
    <TextBox Grid.Row="1" Grid.Column="1"
             PlaceholderText="taro@example.com" />

    <!-- 行2: 送信ボタン（2列にまたがる） -->
    <Button Grid.Row="2" Grid.Column="0" Grid.ColumnSpan="2"
            Content="送信" HorizontalAlignment="Stretch" />
</Grid>
```

---

## 4. 主要コントロール一覧

### 4.1 入力系コントロール

```
+-------------------+------------------------------------------+
| コントロール       | 用途                                      |
+-------------------+------------------------------------------+
| TextBox           | 単一行テキスト入力                        |
| PasswordBox       | パスワード入力（マスク表示）              |
| NumberBox         | 数値入力（増減ボタン付き）                |
| ComboBox          | ドロップダウン選択                        |
| RadioButtons      | 排他選択（グループ化対応）                |
| CheckBox          | 真偽値の切り替え                          |
| ToggleSwitch      | ON/OFF 切り替え                           |
| Slider            | 範囲内の数値選択                          |
| DatePicker        | 日付選択                                  |
| TimePicker        | 時刻選択                                  |
| CalendarDatePicker| カレンダー付き日付選択                    |
| ColorPicker       | 色の選択                                  |
| RatingControl     | 星による評価入力                          |
+-------------------+------------------------------------------+
```

### コード例 4: データバインディングと MVVM

```csharp
// ViewModels/MainViewModel.cs — MVVM パターンの ViewModel
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace MyFirstWinUI.ViewModels;

// ObservableObject を継承して変更通知を自動実装
public partial class MainViewModel : ObservableObject
{
    // [ObservableProperty] で自動的に Name プロパティと
    // PropertyChanged 通知が生成される
    [ObservableProperty]
    private string _name = string.Empty;

    [ObservableProperty]
    private string _greeting = string.Empty;

    // [RelayCommand] でコマンドが自動生成される（GreetCommand）
    [RelayCommand]
    private void Greet()
    {
        Greeting = string.IsNullOrWhiteSpace(Name)
            ? "名前を入力してください"
            : $"こんにちは、{Name} さん！";
    }
}
```

```xml
<!-- MainPage.xaml — ViewModel へのバインディング -->
<Page
    x:Class="MyFirstWinUI.MainPage"
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    xmlns:vm="using:MyFirstWinUI.ViewModels">

    <Page.DataContext>
        <vm:MainViewModel />
    </Page.DataContext>

    <StackPanel Spacing="12" Padding="24">
        <!-- 双方向バインディングで ViewModel の Name と同期 -->
        <TextBox
            Text="{x:Bind ViewModel.Name, Mode=TwoWay, UpdateSourceTrigger=PropertyChanged}"
            PlaceholderText="名前を入力" />

        <!-- コマンドバインディング -->
        <Button
            Content="あいさつ"
            Command="{x:Bind ViewModel.GreetCommand}" />

        <!-- 一方向バインディングで結果を表示 -->
        <TextBlock
            Text="{x:Bind ViewModel.Greeting, Mode=OneWay}"
            Style="{StaticResource SubtitleTextBlockStyle}" />
    </StackPanel>
</Page>
```

---

## 5. スタイルとテーマ

### 5.1 テーマシステム

WinUI 3 は Light / Dark / HighContrast の 3 テーマをネイティブサポートする。

```xml
<!-- App.xaml — テーマリソースの定義 -->
<Application.Resources>
    <ResourceDictionary>
        <ResourceDictionary.MergedDictionaries>
            <!-- WinUI 標準テーマリソース -->
            <XamlControlsResources xmlns="using:Microsoft.UI.Xaml.Controls" />
        </ResourceDictionary.MergedDictionaries>

        <!-- カスタムカラーの定義 -->
        <Color x:Key="BrandColor">#6366F1</Color>
        <SolidColorBrush x:Key="BrandBrush" Color="{StaticResource BrandColor}" />

        <!-- ボタンのスタイルをカスタマイズ -->
        <Style x:Key="BrandButtonStyle" TargetType="Button"
               BasedOn="{StaticResource AccentButtonStyle}">
            <Setter Property="Background" Value="{StaticResource BrandBrush}" />
            <Setter Property="CornerRadius" Value="8" />
            <Setter Property="Padding" Value="16,8" />
        </Style>
    </ResourceDictionary>
</Application.Resources>
```

### 5.2 テーマ切り替えの実装

```csharp
// テーマをプログラムから切り替える
public void SetTheme(ElementTheme theme)
{
    // ルート要素のテーマを変更
    if (App.MainWindow.Content is FrameworkElement rootElement)
    {
        rootElement.RequestedTheme = theme;
    }
}
```

---

## 6. ナビゲーション

### 6.1 NavigationView パターン

```
+-------+----------------------------------+
| ≡     |  ページタイトル            [ _ □ X ] |
+-------+----------------------------------+
| 🏠 ホーム |                                |
| 📊 分析  |     ← ページコンテンツ →         |
| ⚙ 設定  |                                |
|         |                                |
|         |                                |
+-------+----------------------------------+
     ↑                    ↑
  NavigationView      Frame (ページ切替)
```

### コード例 5: NavigationView によるページナビゲーション

```xml
<!-- ShellPage.xaml — ナビゲーションシェル -->
<Page x:Class="MyFirstWinUI.ShellPage"
      xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
      xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml">

    <NavigationView
        x:Name="NavView"
        IsBackButtonVisible="Auto"
        SelectionChanged="NavView_SelectionChanged"
        BackRequested="NavView_BackRequested">

        <!-- ナビゲーション項目の定義 -->
        <NavigationView.MenuItems>
            <NavigationViewItem Content="ホーム" Tag="Home">
                <NavigationViewItem.Icon>
                    <SymbolIcon Symbol="Home" />
                </NavigationViewItem.Icon>
            </NavigationViewItem>

            <NavigationViewItem Content="分析" Tag="Analytics">
                <NavigationViewItem.Icon>
                    <SymbolIcon Symbol="ViewAll" />
                </NavigationViewItem.Icon>
            </NavigationViewItem>
        </NavigationView.MenuItems>

        <!-- 設定ページ（フッター位置に自動配置） -->
        <NavigationView.FooterMenuItems>
            <NavigationViewItem Content="設定" Tag="Settings">
                <NavigationViewItem.Icon>
                    <SymbolIcon Symbol="Setting" />
                </NavigationViewItem.Icon>
            </NavigationViewItem>
        </NavigationView.FooterMenuItems>

        <!-- ページ表示用 Frame -->
        <Frame x:Name="ContentFrame" />
    </NavigationView>
</Page>
```

```csharp
// ShellPage.xaml.cs — ナビゲーションロジック
using Microsoft.UI.Xaml.Controls;

namespace MyFirstWinUI;

public sealed partial class ShellPage : Page
{
    // タグ名とページ型のマッピング辞書
    private readonly Dictionary<string, Type> _pages = new()
    {
        { "Home", typeof(HomePage) },
        { "Analytics", typeof(AnalyticsPage) },
        { "Settings", typeof(SettingsPage) },
    };

    public ShellPage()
    {
        this.InitializeComponent();
        // 初期ページへ遷移
        ContentFrame.Navigate(typeof(HomePage));
    }

    // ナビゲーション項目選択時の処理
    private void NavView_SelectionChanged(
        NavigationView sender,
        NavigationViewSelectionChangedEventArgs args)
    {
        if (args.SelectedItemContainer is NavigationViewItem item
            && item.Tag is string tag
            && _pages.TryGetValue(tag, out var pageType))
        {
            ContentFrame.Navigate(pageType);
        }
    }

    // 戻るボタン押下時の処理
    private void NavView_BackRequested(
        NavigationView sender,
        NavigationViewBackRequestedEventArgs args)
    {
        if (ContentFrame.CanGoBack)
        {
            ContentFrame.GoBack();
        }
    }
}
```

---

## 7. Fluent Design System

### 7.1 Fluent Design の 5 原則

```
+----------------------------------------------------------+
|                  Fluent Design System                     |
+----------------------------------------------------------+
|                                                          |
|  Light        Material       Depth                       |
|  (光)         (素材)        (奥行き)                      |
|  ┌─────┐     ┌─────┐      ┌─────┐                       |
|  │ ░▒▓ │     │ ▒▒▒ │      │ ◈   │                       |
|  │ 照明 │     │ Mica │      │ 影  │                       |
|  └─────┘     └─────┘      └─────┘                       |
|                                                          |
|  Motion                    Scale                         |
|  (動き)                   (適応)                          |
|  ┌─────┐                  ┌─────┐                        |
|  │ → ⟿ │                  │ 📱💻🖥 │                       |
|  │ アニメ│                  │ 応答 │                       |
|  └─────┘                  └─────┘                        |
+----------------------------------------------------------+
```

### 7.2 Mica / Acrylic の適用

```xml
<!-- ウィンドウ背景に Mica を適用 -->
<Window
    x:Class="MyFirstWinUI.MainWindow"
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml">

    <!-- SystemBackdrop で Mica を設定 -->
    <Window.SystemBackdrop>
        <MicaBackdrop />
    </Window.SystemBackdrop>

    <Grid>
        <!-- Acrylic パネル（半透明） -->
        <Grid Background="{ThemeResource AcrylicInAppFillColorDefaultBrush}"
              CornerRadius="8" Padding="16" Margin="24">
            <TextBlock Text="Acrylic 背景のパネル"
                       Style="{StaticResource BodyStrongTextBlockStyle}" />
        </Grid>
    </Grid>
</Window>
```

---

## 8. アンチパターン

### アンチパターン 1: コードビハインドに全ロジックを記述する

```csharp
// NG: コードビハインドにビジネスロジックを直接記述
public sealed partial class OrderPage : Page
{
    private async void SubmitButton_Click(object sender, RoutedEventArgs e)
    {
        // データベース接続からUI更新まで全てここに書く → テスト不能
        var conn = new SqlConnection("...");
        await conn.OpenAsync();
        var cmd = new SqlCommand("INSERT INTO Orders ...", conn);
        await cmd.ExecuteNonQueryAsync();
        ResultText.Text = "注文完了";
    }
}
```

```csharp
// OK: MVVM パターンで ViewModel にロジックを分離
public partial class OrderViewModel : ObservableObject
{
    private readonly IOrderService _orderService;

    [RelayCommand]
    private async Task SubmitOrderAsync()
    {
        // サービス層に委譲 → テスト可能
        await _orderService.CreateOrderAsync(CurrentOrder);
        StatusMessage = "注文完了";
    }
}
```

### アンチパターン 2: UWP 用 API を WinUI 3 でそのまま使う

```csharp
// NG: UWP の名前空間を直接使用（WinUI 3 では動作しない場合がある）
using Windows.UI.Xaml; // ← UWP 用名前空間

// OK: WinUI 3 の名前空間を使用
using Microsoft.UI.Xaml; // ← WinUI 3 用名前空間
```

UWP から移行する際は、名前空間のプレフィックスが `Windows.UI` から `Microsoft.UI` に変更されている点に特に注意が必要である。

---

## 9. FAQ

### Q1: WinUI 3 は .NET MAUI とどう違うのか？

**A:** WinUI 3 は Windows 専用の UI フレームワークであり、Windows のネイティブ機能を最大限に活用できる。一方 .NET MAUI はクロスプラットフォーム（Windows / macOS / iOS / Android）を対象とし、各 OS のネイティブ UI に変換される。Windows のみを対象とし、高品質な UI が必要なら WinUI 3、マルチプラットフォーム展開が必要なら MAUI を選ぶべきである。

### Q2: WinUI 3 アプリは Windows 10 でも動作するか？

**A:** はい。Windows App SDK は Windows 10 バージョン 1809（ビルド 17763）以降をサポートしている。ただし、Mica や SnapLayout など一部の機能は Windows 11 でのみ利用可能である。`ApiInformation.IsApiContractPresent()` を使って機能の存在を確認するのが推奨される。

### Q3: WPF の既存アプリを WinUI 3 に移行するには？

**A:** 完全な自動移行ツールは提供されていない。段階的な移行戦略として、(1) まず MVVM パターンに整理し ViewModel をフレームワーク非依存にする、(2) XAML Islands を使って WPF アプリ内に WinUI 3 コントロールを埋め込む、(3) 最終的にアプリ全体を WinUI 3 で再構築する、というアプローチが推奨される。

---

## 10. まとめ

| トピック | キーポイント |
|---|---|
| WinUI 3 の位置づけ | Windows App SDK の UI 層。WPF の後継として新規開発に推奨 |
| プロジェクト作成 | Visual Studio テンプレート + Windows App SDK NuGet |
| XAML | 宣言的 UI 記述。Grid / StackPanel でレイアウト構築 |
| データバインディング | `x:Bind` による型安全なバインディングが推奨 |
| MVVM | CommunityToolkit.Mvvm でボイラープレートを削減 |
| スタイル・テーマ | Light / Dark テーマ標準対応。リソースディクショナリで管理 |
| ナビゲーション | NavigationView + Frame パターンが標準 |
| Fluent Design | Mica / Acrylic / アニメーション で現代的な外観を実現 |

---

## 次に読むべきガイド

- **[02-webview2.md](./02-webview2.md)** — WebView2 を統合してハイブリッドアプリを構築する方法
- **パッケージングと署名** — MSIX パッケージによる配布方法

---

## 参考文献

1. Microsoft, "Windows App SDK — WinUI 3", https://learn.microsoft.com/windows/apps/winui/winui3/
2. Microsoft, "XAML Controls Gallery", https://github.com/microsoft/Xaml-Controls-Gallery
3. Microsoft, "CommunityToolkit.Mvvm", https://learn.microsoft.com/dotnet/communitytoolkit/mvvm/
4. Microsoft, "Fluent Design System", https://fluent2.microsoft.design/

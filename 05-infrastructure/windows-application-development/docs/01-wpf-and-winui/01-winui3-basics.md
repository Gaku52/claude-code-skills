# WinUI 3 の基本

> Windows App SDK に含まれる最新の UI フレームワーク WinUI 3 を使い、Fluent Design に準拠したデスクトップアプリケーションを構築する方法を体系的に学ぶ。

---

## この章で学ぶこと

1. **WinUI 3 プロジェクトの作成**からビルド・実行までの一連のワークフローを理解する
2. **XAML の基礎構文**とデータバインディング、主要コントロールの使い方を習得する
3. **Fluent Design System** のスタイル・テーマ・ナビゲーションパターンを実装できるようになる
4. **依存性注入**と **MVVM パターン**を活用した保守性の高いアプリ設計を習得する
5. **ContentDialog** や **TeachingTip** など WinUI 3 固有のコントロールを使いこなせるようになる

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

### 1.3 Windows App SDK のバージョンと機能

```
Windows App SDK バージョン履歴:

  1.0 (2021-11) ─── 初回安定版リリース
                    ・WinUI 3 コントロール基本セット
                    ・AppWindow API
                    ・MRT (リソース管理)

  1.1 (2022-03) ─── 機能強化
                    ・Mica 背景サポート
                    ・Self-contained デプロイ
                    ・環境マネージャ

  1.2 (2022-08) ─── パフォーマンス改善
                    ・AppNotification API
                    ・ウィジェットサポート
                    ・改善された AppWindow

  1.3 (2023-02) ─── 安定性向上
                    ・MSIX 不要のデプロイ改善
                    ・新しい MapControl

  1.4 (2023-08) ─── 最新
                    ・改善された ItemsView
                    ・WebView2 の改善
                    ・新しい AnnotatedScrollBar

  1.5+ (2024)  ─── 継続的改善
                    ・パフォーマンス最適化
                    ・.NET 8 対応の強化
```

---

## 2. プロジェクトの作成

### 2.1 前提条件

- Visual Studio 2022 17.8 以降
- Windows App SDK 拡張機能（NuGet: `Microsoft.WindowsAppSDK`）
- .NET 8 SDK 以降
- Windows 10 バージョン 1809 (ビルド 17763) 以降

### 2.2 テンプレートからの作成

```
Visual Studio → 新しいプロジェクトの作成
  → "Blank App, Packaged (WinUI 3 in Desktop)" を選択
  → プロジェクト名: MyFirstWinUI
  → ターゲットフレームワーク: net8.0-windows10.0.19041.0
```

### 2.3 プロジェクト構成

```
MyFirstWinUI/
├── MyFirstWinUI.csproj           ← プロジェクト設定
├── app.manifest                  ← アプリマニフェスト（DPI 対応等）
├── Package.appxmanifest          ← MSIX パッケージ設定
├── App.xaml                      ← アプリ共通リソース定義
├── App.xaml.cs                   ← アプリエントリポイント
├── MainWindow.xaml               ← メインウィンドウの XAML
├── MainWindow.xaml.cs            ← メインウィンドウのコードビハインド
├── Assets/                       ← アイコン・スプラッシュ画像
│   ├── LockScreenLogo.png
│   ├── SplashScreen.png
│   ├── Square44x44Logo.png
│   ├── Square150x150Logo.png
│   ├── StoreLogo.png
│   └── Wide310x150Logo.png
├── ViewModels/                   ← ViewModel 層
│   └── MainViewModel.cs
├── Views/                        ← ページ（View）層
│   ├── HomePage.xaml
│   ├── HomePage.xaml.cs
│   ├── SettingsPage.xaml
│   └── SettingsPage.xaml.cs
├── Models/                       ← モデル層
│   └── AppConfig.cs
├── Services/                     ← サービス層
│   ├── INavigationService.cs
│   └── NavigationService.cs
└── Helpers/                      ← ユーティリティ
    └── WindowHelper.cs
```

### 2.4 csproj の設定

```xml
<!-- MyFirstWinUI.csproj -->
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>WinExe</OutputType>
    <TargetFramework>net8.0-windows10.0.19041.0</TargetFramework>
    <TargetPlatformMinVersion>10.0.17763.0</TargetPlatformMinVersion>
    <RootNamespace>MyFirstWinUI</RootNamespace>
    <ApplicationManifest>app.manifest</ApplicationManifest>
    <Platforms>x86;x64;ARM64</Platforms>
    <RuntimeIdentifiers>win-x86;win-x64;win-arm64</RuntimeIdentifiers>
    <UseWinUI>true</UseWinUI>
    <WindowsSdkPackageVersion>10.0.19041.38</WindowsSdkPackageVersion>
    <!-- Nullable 参照型を有効化 -->
    <Nullable>enable</Nullable>
    <!-- トリミング対応（Self-contained デプロイ時） -->
    <PublishTrimmed>true</PublishTrimmed>
    <TrimMode>partial</TrimMode>
  </PropertyGroup>

  <ItemGroup>
    <!-- Windows App SDK -->
    <PackageReference Include="Microsoft.WindowsAppSDK" Version="1.5.240607001" />
    <!-- CommunityToolkit.Mvvm -->
    <PackageReference Include="CommunityToolkit.Mvvm" Version="8.2.2" />
    <!-- DI コンテナ -->
    <PackageReference Include="Microsoft.Extensions.DependencyInjection" Version="8.0.0" />
    <!-- WinUI Community Toolkit -->
    <PackageReference Include="CommunityToolkit.WinUI.UI.Controls" Version="7.1.2" />
  </ItemGroup>
</Project>
```

### コード例 1: App.xaml.cs ― アプリケーションエントリポイント

```csharp
// App.xaml.cs — アプリケーションのエントリポイント
using Microsoft.Extensions.DependencyInjection;
using Microsoft.UI.Xaml;

namespace MyFirstWinUI;

public partial class App : Application
{
    private Window? _window;

    // DI コンテナ
    public IServiceProvider Services { get; }
    public static new App Current => (App)Application.Current;

    public App()
    {
        this.InitializeComponent();

        // サービスの登録
        var services = new ServiceCollection();
        ConfigureServices(services);
        Services = services.BuildServiceProvider();
    }

    private static void ConfigureServices(IServiceCollection services)
    {
        // ViewModel の登録
        services.AddTransient<ViewModels.MainViewModel>();
        services.AddTransient<ViewModels.SettingsViewModel>();

        // サービスの登録
        services.AddSingleton<Services.INavigationService, Services.NavigationService>();
        services.AddSingleton<Services.IThemeService, Services.ThemeService>();
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

        // ウィンドウサイズの設定
        var appWindow = this.AppWindow;
        appWindow.Resize(new Windows.Graphics.SizeInt32(1200, 800));

        // タイトルバーのカスタマイズ
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(null); // デフォルトのドラッグ可能領域を使用
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
<Window>                          -- ルート要素
  +-- <StackPanel>                -- レイアウトパネル
  |   +-- <TextBlock Text="..." />  -- コンテンツ要素
  |   +-- <Button Content="..." />  -- インタラクティブ要素
  |   +-- <Image Source="..." />    -- メディア要素
  +-- <Window.Resources>           -- リソース定義
      +-- <Style TargetType="..." /> -- スタイル
```

### 3.2 レイアウトパネルの比較

| パネル | 配置方式 | 主な用途 |
|---|---|---|
| `StackPanel` | 水平 or 垂直に直列配置 | 単純なフォーム、ツールバー |
| `Grid` | 行と列のセル配置 | 複雑なレイアウト全般 |
| `RelativePanel` | 相対位置指定 | レスポンシブ配置 |
| `Canvas` | 絶対座標指定 | 描画系、ドラッグ操作 |
| `WrapPanel`* | 折り返し配置 | タグ一覧、サムネイル |
| `UniformGrid`* | 均等配置 | カレンダー、ボタングリッド |

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

### 3.3 マージンとパディングの設定

```xml
<!-- マージン・パディングの記法 -->

<!-- 全方向同じ値 -->
<Button Margin="16" Padding="8" Content="ボタン" />

<!-- 水平, 垂直 -->
<Button Margin="16,8" Content="ボタン" />

<!-- 左, 上, 右, 下（時計回り） -->
<Button Margin="16,8,16,24" Content="ボタン" />

<!-- Alignment の組み合わせ -->
<StackPanel HorizontalAlignment="Center"
            VerticalAlignment="Top"
            Margin="0,24,0,0">
    <TextBlock Text="中央上部に配置"
               TextAlignment="Center" />
</StackPanel>
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
| AutoSuggestBox    | オートコンプリート付きテキスト入力         |
+-------------------+------------------------------------------+
```

### 4.2 表示系コントロール

```xml
<!-- InfoBar: 情報バー（成功・警告・エラーメッセージの表示） -->
<InfoBar
    Title="保存完了"
    Message="設定が正常に保存されました。"
    Severity="Success"
    IsOpen="{x:Bind ViewModel.ShowSuccessBar, Mode=OneWay}" />

<InfoBar
    Title="エラー"
    Message="ネットワーク接続を確認してください。"
    Severity="Error"
    IsOpen="True"
    IsClosable="True" />

<!-- ProgressBar: 進捗バー -->
<ProgressBar Value="{x:Bind ViewModel.Progress, Mode=OneWay}"
             Maximum="100" />

<!-- 不確定な進捗（ロード中） -->
<ProgressBar IsIndeterminate="True" />

<!-- ProgressRing: ローディングスピナー -->
<ProgressRing IsActive="{x:Bind ViewModel.IsLoading, Mode=OneWay}" />

<!-- Expander: 展開可能なパネル -->
<Expander Header="詳細設定" IsExpanded="False">
    <StackPanel Spacing="8">
        <TextBox Header="API キー" />
        <NumberBox Header="タイムアウト (秒)" Value="30" />
    </StackPanel>
</Expander>

<!-- TeachingTip: ツールチップ型のガイダンス -->
<TeachingTip
    x:Name="SaveTip"
    Target="{x:Bind SaveButton}"
    Title="自動保存が有効です"
    Subtitle="変更は自動的に保存されます。手動で保存する必要はありません。"
    PreferredPlacement="Bottom"
    IsLightDismissEnabled="True" />

<!-- Breadcrumb: パンくずリスト -->
<BreadcrumbBar ItemsSource="{x:Bind ViewModel.BreadcrumbItems}">
    <BreadcrumbBar.ItemTemplate>
        <DataTemplate x:DataType="x:String">
            <TextBlock Text="{x:Bind}" />
        </DataTemplate>
    </BreadcrumbBar.ItemTemplate>
</BreadcrumbBar>
```

### 4.3 コレクション表示コントロール

```xml
<!-- ListView: 垂直リスト -->
<ListView ItemsSource="{x:Bind ViewModel.Tasks, Mode=OneWay}"
          SelectionMode="Single"
          SelectedItem="{x:Bind ViewModel.SelectedTask, Mode=TwoWay}">
    <ListView.ItemTemplate>
        <DataTemplate x:DataType="models:TaskItem">
            <Grid Padding="8" ColumnSpacing="12">
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="Auto" />
                    <ColumnDefinition Width="*" />
                    <ColumnDefinition Width="Auto" />
                </Grid.ColumnDefinitions>

                <CheckBox Grid.Column="0"
                          IsChecked="{x:Bind IsCompleted, Mode=TwoWay}" />
                <StackPanel Grid.Column="1">
                    <TextBlock Text="{x:Bind Title}"
                               Style="{StaticResource BodyStrongTextBlockStyle}" />
                    <TextBlock Text="{x:Bind Description}"
                               Style="{StaticResource CaptionTextBlockStyle}"
                               Opacity="0.7" />
                </StackPanel>
                <TextBlock Grid.Column="2"
                           Text="{x:Bind Priority}"
                           VerticalAlignment="Center" />
            </Grid>
        </DataTemplate>
    </ListView.ItemTemplate>
</ListView>

<!-- GridView: グリッド表示 -->
<GridView ItemsSource="{x:Bind ViewModel.Images, Mode=OneWay}"
          SelectionMode="Multiple"
          IsItemClickEnabled="True"
          ItemClick="GridView_ItemClick">
    <GridView.ItemTemplate>
        <DataTemplate x:DataType="models:ImageItem">
            <Grid Width="200" Height="200" CornerRadius="8">
                <Image Source="{x:Bind ThumbnailUrl}"
                       Stretch="UniformToFill" />
                <TextBlock Text="{x:Bind FileName}"
                           VerticalAlignment="Bottom"
                           Padding="8"
                           Background="{ThemeResource AcrylicInAppFillColorDefaultBrush}"
                           Foreground="{ThemeResource TextFillColorPrimaryBrush}" />
            </Grid>
        </DataTemplate>
    </GridView.ItemTemplate>
</GridView>
```

### コード例 4: データバインディングと MVVM

```csharp
// ViewModels/MainViewModel.cs — MVVM パターンの ViewModel
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Collections.ObjectModel;

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

    [ObservableProperty]
    private bool _isLoading;

    [ObservableProperty]
    private ObservableCollection<TaskItem> _tasks = new();

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(DeleteTaskCommand))]
    private TaskItem? _selectedTask;

    // [RelayCommand] でコマンドが自動生成される（GreetCommand）
    [RelayCommand]
    private void Greet()
    {
        Greeting = string.IsNullOrWhiteSpace(Name)
            ? "名前を入力してください"
            : $"こんにちは、{Name} さん！";
    }

    [RelayCommand]
    private async Task LoadTasksAsync()
    {
        IsLoading = true;
        try
        {
            // 非同期でデータを取得
            await Task.Delay(500); // シミュレーション
            Tasks = new ObservableCollection<TaskItem>(
                new[]
                {
                    new TaskItem { Title = "設計レビュー", Priority = "高", IsCompleted = false },
                    new TaskItem { Title = "テスト作成", Priority = "中", IsCompleted = true },
                    new TaskItem { Title = "ドキュメント更新", Priority = "低", IsCompleted = false },
                });
        }
        finally
        {
            IsLoading = false;
        }
    }

    private bool CanDeleteTask() => SelectedTask != null;

    [RelayCommand(CanExecute = nameof(CanDeleteTask))]
    private void DeleteTask()
    {
        if (SelectedTask != null)
        {
            Tasks.Remove(SelectedTask);
            SelectedTask = null;
        }
    }
}

public class TaskItem
{
    public string Title { get; set; } = "";
    public string Description { get; set; } = "";
    public string Priority { get; set; } = "中";
    public bool IsCompleted { get; set; }
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

        <!-- カスタムリソースディクショナリの読み込み -->
        <!-- <ResourceDictionary Source="Styles/CustomStyles.xaml" /> -->
    </ResourceDictionary>
</Application.Resources>
```

### 5.2 テーマ切り替えの実装

```csharp
// テーマサービスの実装
using Microsoft.UI.Xaml;

namespace MyFirstWinUI.Services;

public interface IThemeService
{
    ElementTheme CurrentTheme { get; }
    void SetTheme(ElementTheme theme);
    void ToggleTheme();
}

public class ThemeService : IThemeService
{
    private ElementTheme _currentTheme = ElementTheme.Default;

    public ElementTheme CurrentTheme => _currentTheme;

    public void SetTheme(ElementTheme theme)
    {
        _currentTheme = theme;

        // ルート要素のテーマを変更
        if (App.MainWindow?.Content is FrameworkElement rootElement)
        {
            rootElement.RequestedTheme = theme;
        }
    }

    public void ToggleTheme()
    {
        var currentTheme = _currentTheme;
        if (currentTheme == ElementTheme.Dark)
        {
            SetTheme(ElementTheme.Light);
        }
        else
        {
            SetTheme(ElementTheme.Dark);
        }
    }
}
```

### 5.3 カスタムスタイルの詳細

```xml
<!-- Styles/CustomStyles.xaml — カスタムスタイル集 -->
<ResourceDictionary
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml">

    <!-- カード風パネルのスタイル -->
    <Style x:Key="CardStyle" TargetType="Border">
        <Setter Property="Background"
                Value="{ThemeResource CardBackgroundFillColorDefaultBrush}" />
        <Setter Property="BorderBrush"
                Value="{ThemeResource CardStrokeColorDefaultBrush}" />
        <Setter Property="BorderThickness" Value="1" />
        <Setter Property="CornerRadius" Value="8" />
        <Setter Property="Padding" Value="16" />
    </Style>

    <!-- セクションヘッダーのスタイル -->
    <Style x:Key="SectionHeaderStyle" TargetType="TextBlock"
           BasedOn="{StaticResource SubtitleTextBlockStyle}">
        <Setter Property="Margin" Value="0,24,0,8" />
    </Style>

    <!-- 設定項目のスタイル -->
    <Style x:Key="SettingItemStyle" TargetType="Grid">
        <Setter Property="Padding" Value="16" />
        <Setter Property="Background"
                Value="{ThemeResource CardBackgroundFillColorDefaultBrush}" />
        <Setter Property="CornerRadius" Value="4" />
        <Setter Property="Margin" Value="0,2" />
    </Style>

    <!-- サブテキストのスタイル -->
    <Style x:Key="SubtextStyle" TargetType="TextBlock"
           BasedOn="{StaticResource CaptionTextBlockStyle}">
        <Setter Property="Foreground"
                Value="{ThemeResource TextFillColorSecondaryBrush}" />
        <Setter Property="TextWrapping" Value="Wrap" />
    </Style>
</ResourceDictionary>
```

---

## 6. ナビゲーション

### 6.1 NavigationView パターン

```
+-------+----------------------------------+
| =     |  ページタイトル            [ _ # X ] |
+-------+----------------------------------+
| Home  |                                |
| 分析   |     <-- ページコンテンツ -->       |
| 設定   |                                |
|         |                                |
|         |                                |
+-------+----------------------------------+
     ^                    ^
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
        BackRequested="NavView_BackRequested"
        PaneDisplayMode="Left"
        IsPaneToggleButtonVisible="True"
        IsSettingsVisible="True">

        <!-- ヘッダー -->
        <NavigationView.AutoSuggestBox>
            <AutoSuggestBox PlaceholderText="検索..."
                            QueryIcon="Find" />
        </NavigationView.AutoSuggestBox>

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

            <NavigationViewItemSeparator />

            <NavigationViewItemHeader Content="管理" />

            <NavigationViewItem Content="ユーザー" Tag="Users">
                <NavigationViewItem.Icon>
                    <SymbolIcon Symbol="People" />
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
using Microsoft.UI.Xaml.Navigation;

namespace MyFirstWinUI;

public sealed partial class ShellPage : Page
{
    // タグ名とページ型のマッピング辞書
    private readonly Dictionary<string, Type> _pages = new()
    {
        { "Home", typeof(HomePage) },
        { "Analytics", typeof(AnalyticsPage) },
        { "Users", typeof(UsersPage) },
        { "Settings", typeof(SettingsPage) },
    };

    public ShellPage()
    {
        this.InitializeComponent();

        // 初期ページへ遷移
        ContentFrame.Navigate(typeof(HomePage));

        // フレームのナビゲーション完了イベント
        ContentFrame.Navigated += ContentFrame_Navigated;
    }

    // ナビゲーション項目選択時の処理
    private void NavView_SelectionChanged(
        NavigationView sender,
        NavigationViewSelectionChangedEventArgs args)
    {
        if (args.IsSettingsSelected)
        {
            // 設定ページへ遷移
            ContentFrame.Navigate(typeof(SettingsPage));
            return;
        }

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

    // ナビゲーション完了時にナビゲーション項目のハイライトを同期
    private void ContentFrame_Navigated(object sender, NavigationEventArgs e)
    {
        // 戻るボタンの表示/非表示
        NavView.IsBackEnabled = ContentFrame.CanGoBack;

        // 現在のページに対応するナビゲーション項目を選択
        var pageType = ContentFrame.CurrentSourcePageType;
        var tag = _pages.FirstOrDefault(p => p.Value == pageType).Key;

        if (tag != null)
        {
            foreach (NavigationViewItem item in NavView.MenuItems.OfType<NavigationViewItem>())
            {
                if (item.Tag?.ToString() == tag)
                {
                    NavView.SelectedItem = item;
                    break;
                }
            }
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
|  +-----+     +-----+      +-----+                       |
|  | ### |     | ### |      |  *  |                       |
|  | 照明 |     | Mica |      | 影  |                       |
|  +-----+     +-----+      +-----+                       |
|                                                          |
|  Motion                    Scale                         |
|  (動き)                   (適応)                          |
|  +-----+                  +-----+                        |
|  | --> |                  | [ ] |                        |
|  |アニメ|                  | 応答 |                        |
|  +-----+                  +-----+                        |
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

### 7.3 アニメーションの実装

```csharp
// Composition API を使ったアニメーション
using Microsoft.UI.Composition;
using Microsoft.UI.Xaml.Hosting;

public static class AnimationHelper
{
    // フェードインアニメーション
    public static void FadeIn(UIElement element, TimeSpan duration)
    {
        var visual = ElementCompositionPreview.GetElementVisual(element);
        var compositor = visual.Compositor;

        var animation = compositor.CreateScalarKeyFrameAnimation();
        animation.InsertKeyFrame(0f, 0f);
        animation.InsertKeyFrame(1f, 1f);
        animation.Duration = duration;

        visual.StartAnimation("Opacity", animation);
    }

    // スライドインアニメーション
    public static void SlideIn(UIElement element, TimeSpan duration, float offsetX = 0, float offsetY = 50)
    {
        var visual = ElementCompositionPreview.GetElementVisual(element);
        var compositor = visual.Compositor;

        // オフセットアニメーション
        var offsetAnimation = compositor.CreateVector3KeyFrameAnimation();
        offsetAnimation.InsertKeyFrame(0f, new System.Numerics.Vector3(offsetX, offsetY, 0));
        offsetAnimation.InsertKeyFrame(1f, new System.Numerics.Vector3(0, 0, 0));
        offsetAnimation.Duration = duration;

        // フェードアニメーション
        var fadeAnimation = compositor.CreateScalarKeyFrameAnimation();
        fadeAnimation.InsertKeyFrame(0f, 0f);
        fadeAnimation.InsertKeyFrame(1f, 1f);
        fadeAnimation.Duration = duration;

        visual.StartAnimation("Offset", offsetAnimation);
        visual.StartAnimation("Opacity", fadeAnimation);
    }

    // スケールアニメーション
    public static void ScaleIn(UIElement element, TimeSpan duration)
    {
        var visual = ElementCompositionPreview.GetElementVisual(element);
        var compositor = visual.Compositor;

        // 中央からスケール
        visual.CenterPoint = new System.Numerics.Vector3(
            (float)(element as FrameworkElement)?.ActualWidth / 2 ?? 0,
            (float)(element as FrameworkElement)?.ActualHeight / 2 ?? 0,
            0);

        var scaleAnimation = compositor.CreateVector3KeyFrameAnimation();
        scaleAnimation.InsertKeyFrame(0f, new System.Numerics.Vector3(0.8f, 0.8f, 1f));
        scaleAnimation.InsertKeyFrame(1f, new System.Numerics.Vector3(1f, 1f, 1f));
        scaleAnimation.Duration = duration;

        visual.StartAnimation("Scale", scaleAnimation);
    }
}
```

```xml
<!-- XAML でのアニメーション（Storyboard） -->
<Page.Resources>
    <Storyboard x:Key="FadeInStoryboard">
        <DoubleAnimation
            Storyboard.TargetName="ContentPanel"
            Storyboard.TargetProperty="Opacity"
            From="0" To="1" Duration="0:0:0.5">
            <DoubleAnimation.EasingFunction>
                <CubicEase EasingMode="EaseOut" />
            </DoubleAnimation.EasingFunction>
        </DoubleAnimation>
    </Storyboard>
</Page.Resources>

<!-- EntranceThemeTransition: ページ遷移時のアニメーション -->
<StackPanel x:Name="ContentPanel">
    <StackPanel.ChildrenTransitions>
        <EntranceThemeTransition IsStaggeringEnabled="True" />
    </StackPanel.ChildrenTransitions>
    <!-- 子要素が順番にフェードインする -->
    <TextBlock Text="項目1" />
    <TextBlock Text="項目2" />
    <TextBlock Text="項目3" />
</StackPanel>
```

---

## 8. ContentDialog の実装

```csharp
// ContentDialog: モーダルダイアログの実装
using Microsoft.UI.Xaml.Controls;

public static class DialogHelper
{
    // 確認ダイアログ
    public static async Task<bool> ShowConfirmAsync(
        XamlRoot xamlRoot,
        string title,
        string content)
    {
        var dialog = new ContentDialog
        {
            Title = title,
            Content = content,
            PrimaryButtonText = "はい",
            SecondaryButtonText = "キャンセル",
            DefaultButton = ContentDialogButton.Primary,
            XamlRoot = xamlRoot,
        };

        var result = await dialog.ShowAsync();
        return result == ContentDialogResult.Primary;
    }

    // カスタムコンテンツのダイアログ
    public static async Task<string?> ShowInputDialogAsync(
        XamlRoot xamlRoot,
        string title,
        string placeholder = "")
    {
        var inputBox = new TextBox
        {
            PlaceholderText = placeholder,
            AcceptsReturn = false,
        };

        var dialog = new ContentDialog
        {
            Title = title,
            Content = inputBox,
            PrimaryButtonText = "OK",
            SecondaryButtonText = "キャンセル",
            DefaultButton = ContentDialogButton.Primary,
            XamlRoot = xamlRoot,
        };

        var result = await dialog.ShowAsync();
        return result == ContentDialogResult.Primary
            ? inputBox.Text
            : null;
    }
}
```

```xml
<!-- XAML で定義する ContentDialog -->
<ContentDialog
    x:Name="DeleteConfirmDialog"
    Title="タスクの削除"
    PrimaryButtonText="削除"
    SecondaryButtonText="キャンセル"
    DefaultButton="Secondary"
    PrimaryButtonClick="DeleteConfirmDialog_PrimaryButtonClick">
    <StackPanel Spacing="8">
        <TextBlock Text="本当にこのタスクを削除しますか？" />
        <TextBlock Text="この操作は取り消せません。"
                   Style="{StaticResource CaptionTextBlockStyle}"
                   Foreground="{ThemeResource SystemFillColorCriticalBrush}" />
    </StackPanel>
</ContentDialog>
```

---

## 9. ウィンドウ管理（AppWindow API）

```csharp
// AppWindow を使ったウィンドウの高度な制御
using Microsoft.UI;
using Microsoft.UI.Windowing;
using Windows.Graphics;

public sealed partial class MainWindow : Window
{
    private AppWindow _appWindow;

    public MainWindow()
    {
        this.InitializeComponent();

        // AppWindow の取得
        _appWindow = this.AppWindow;

        // ウィンドウサイズの設定
        _appWindow.Resize(new SizeInt32(1200, 800));

        // 最小サイズの設定
        _appWindow.Changed += (sender, args) =>
        {
            if (args.DidSizeChange)
            {
                var size = sender.Size;
                if (size.Width < 800 || size.Height < 600)
                {
                    sender.Resize(new SizeInt32(
                        Math.Max(size.Width, 800),
                        Math.Max(size.Height, 600)));
                }
            }
        };

        // タイトルバーの色をカスタマイズ
        if (AppWindowTitleBar.IsCustomizationSupported())
        {
            var titleBar = _appWindow.TitleBar;
            titleBar.ExtendsContentIntoTitleBar = true;
            titleBar.ButtonBackgroundColor = Colors.Transparent;
            titleBar.ButtonInactiveBackgroundColor = Colors.Transparent;
            titleBar.ButtonHoverBackgroundColor = Windows.UI.Color.FromArgb(25, 255, 255, 255);
        }

        // フルスクリーンの切り替え
        // _appWindow.SetPresenter(AppWindowPresenterKind.FullScreen);

        // コンパクトオーバーレイ（ピクチャ・イン・ピクチャ風）
        // _appWindow.SetPresenter(AppWindowPresenterKind.CompactOverlay);
    }

    // ウィンドウを画面中央に配置
    private void CenterWindow()
    {
        var displayArea = DisplayArea.GetFromWindowId(
            _appWindow.Id, DisplayAreaFallback.Nearest);
        var workArea = displayArea.WorkArea;
        var windowSize = _appWindow.Size;

        _appWindow.Move(new PointInt32(
            (workArea.Width - windowSize.Width) / 2 + workArea.X,
            (workArea.Height - windowSize.Height) / 2 + workArea.Y));
    }
}
```

---

## 10. アンチパターン

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

    public OrderViewModel(IOrderService orderService)
    {
        _orderService = orderService;
    }

    [ObservableProperty]
    private string _statusMessage = "";

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
using Windows.UI.Xaml; // UWP 用名前空間

// OK: WinUI 3 の名前空間を使用
using Microsoft.UI.Xaml; // WinUI 3 用名前空間
```

UWP から移行する際は、名前空間のプレフィックスが `Windows.UI` から `Microsoft.UI` に変更されている点に特に注意が必要である。

### アンチパターン 3: UI スレッドのブロック

```csharp
// NG: UI スレッドで同期的に重い処理を実行
private void LoadData_Click(object sender, RoutedEventArgs e)
{
    // Thread.Sleep や同期 I/O は UI をフリーズさせる
    Thread.Sleep(3000);
    var data = File.ReadAllText("large-file.txt"); // 同期 I/O
    DataText.Text = data;
}
```

```csharp
// OK: 非同期処理で UI スレッドをブロックしない
private async void LoadData_Click(object sender, RoutedEventArgs e)
{
    LoadingRing.IsActive = true;
    LoadButton.IsEnabled = false;

    try
    {
        // 非同期 I/O で UI をブロックしない
        var data = await File.ReadAllTextAsync("large-file.txt");

        // DispatcherQueue を使って UI スレッドで更新
        DispatcherQueue.TryEnqueue(() =>
        {
            DataText.Text = data;
        });
    }
    catch (Exception ex)
    {
        // エラー表示
        ErrorBar.Message = ex.Message;
        ErrorBar.IsOpen = true;
    }
    finally
    {
        LoadingRing.IsActive = false;
        LoadButton.IsEnabled = true;
    }
}
```

---

## 11. テストの実装

```csharp
// ViewModel のユニットテスト
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;

[TestClass]
public class MainViewModelTests
{
    [TestMethod]
    public void Greet_WithName_ReturnsGreeting()
    {
        // Arrange
        var viewModel = new MainViewModel();
        viewModel.Name = "太郎";

        // Act
        viewModel.GreetCommand.Execute(null);

        // Assert
        Assert.AreEqual("こんにちは、太郎 さん！", viewModel.Greeting);
    }

    [TestMethod]
    public void Greet_WithEmptyName_ReturnsPrompt()
    {
        var viewModel = new MainViewModel();
        viewModel.Name = "";

        viewModel.GreetCommand.Execute(null);

        Assert.AreEqual("名前を入力してください", viewModel.Greeting);
    }

    [TestMethod]
    public async Task LoadTasks_SetsIsLoading()
    {
        var viewModel = new MainViewModel();

        var loadTask = viewModel.LoadTasksCommand.ExecuteAsync(null);

        // LoadTasks 実行中は IsLoading が true
        Assert.IsTrue(viewModel.IsLoading);

        await loadTask;

        // 完了後は false
        Assert.IsFalse(viewModel.IsLoading);
    }

    [TestMethod]
    public void DeleteTask_WithSelection_RemovesTask()
    {
        var viewModel = new MainViewModel();
        var task = new TaskItem { Title = "テスト" };
        viewModel.Tasks.Add(task);
        viewModel.SelectedTask = task;

        viewModel.DeleteTaskCommand.Execute(null);

        Assert.AreEqual(0, viewModel.Tasks.Count);
        Assert.IsNull(viewModel.SelectedTask);
    }

    [TestMethod]
    public void DeleteTask_WithoutSelection_CannotExecute()
    {
        var viewModel = new MainViewModel();
        viewModel.SelectedTask = null;

        Assert.IsFalse(viewModel.DeleteTaskCommand.CanExecute(null));
    }
}
```

---

## 12. FAQ

### Q1: WinUI 3 は .NET MAUI とどう違うのか？

**A:** WinUI 3 は Windows 専用の UI フレームワークであり、Windows のネイティブ機能を最大限に活用できる。一方 .NET MAUI はクロスプラットフォーム（Windows / macOS / iOS / Android）を対象とし、各 OS のネイティブ UI に変換される。Windows のみを対象とし、高品質な UI が必要なら WinUI 3、マルチプラットフォーム展開が必要なら MAUI を選ぶべきである。

### Q2: WinUI 3 アプリは Windows 10 でも動作するか？

**A:** はい。Windows App SDK は Windows 10 バージョン 1809（ビルド 17763）以降をサポートしている。ただし、Mica や SnapLayout など一部の機能は Windows 11 でのみ利用可能である。`ApiInformation.IsApiContractPresent()` を使って機能の存在を確認するのが推奨される。

### Q3: WPF の既存アプリを WinUI 3 に移行するには？

**A:** 完全な自動移行ツールは提供されていない。段階的な移行戦略として、(1) まず MVVM パターンに整理し ViewModel をフレームワーク非依存にする、(2) XAML Islands を使って WPF アプリ内に WinUI 3 コントロールを埋め込む、(3) 最終的にアプリ全体を WinUI 3 で再構築する、というアプローチが推奨される。

### Q4: WinUI 3 で WebView2 を使うには？

**A:** NuGet パッケージ `Microsoft.Web.WebView2` をインストールし、XAML に `<WebView2 Source="https://example.com" />` を配置する。WebView2 は Chromium ベースのブラウザコントロールであり、Edge (Chromium) の Evergreen ランタイムを使用する。JavaScript との双方向通信も可能で、ハイブリッドアプリの構築に適している。

### Q5: WinUI 3 の MSIX パッケージと非パッケージの違いは？

**A:** MSIX パッケージではクリーンなインストール/アンインストール、自動更新、Windows Store 配布が可能である。非パッケージ（Unpackaged）では従来の exe 配布と同様に自由な配布が可能で、レジストリやファイルシステムへのフルアクセスが得られる。新規プロジェクトでは MSIX パッケージが推奨されるが、既存の配布インフラとの互換性が必要な場合は非パッケージを選択する。

---

## 13. まとめ

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
| ダイアログ | ContentDialog でモーダル UI を実装 |
| ウィンドウ管理 | AppWindow API でサイズ・位置・プレゼンターを制御 |
| テスト | ViewModel のユニットテストで品質を確保 |

---

## 次に読むべきガイド

- **[02-webview2.md](./02-webview2.md)** -- WebView2 を統合してハイブリッドアプリを構築する方法
- **パッケージングと署名** -- MSIX パッケージによる配布方法

---

## 参考文献

1. Microsoft, "Windows App SDK -- WinUI 3", https://learn.microsoft.com/windows/apps/winui/winui3/
2. Microsoft, "XAML Controls Gallery", https://github.com/microsoft/Xaml-Controls-Gallery
3. Microsoft, "CommunityToolkit.Mvvm", https://learn.microsoft.com/dotnet/communitytoolkit/mvvm/
4. Microsoft, "Fluent Design System", https://fluent2.microsoft.design/
5. Microsoft, "AppWindow Class", https://learn.microsoft.com/windows/windows-app-sdk/api/winrt/microsoft.ui.windowing.appwindow
6. Microsoft, "Windows App SDK Release Notes", https://learn.microsoft.com/windows/apps/windows-app-sdk/stable-channel

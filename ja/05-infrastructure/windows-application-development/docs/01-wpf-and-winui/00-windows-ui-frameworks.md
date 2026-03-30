# Windows UI フレームワーク比較

> Windows ネイティブ UI フレームワークは WPF から WinUI 3 へ進化した。WPF、WinUI 3、UWP、MAUI の歴史的経緯と使い分け、XAML の基礎、データバインディング、MVVM パターンまで解説する。

## この章で学ぶこと

- [ ] Windows UI フレームワークの歴史と選定基準を理解する
- [ ] XAML の基礎構文を把握する
- [ ] MVVM パターンとデータバインディングを理解する
- [ ] 各フレームワークの具体的なコード例を通じて実装差異を理解する
- [ ] プロジェクトの要件に応じた適切なフレームワーク選定ができる


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

---

## 1. フレームワークの歴史と比較

### 1.1 Windows UI フレームワークの進化

```
Windows UI フレームワークの進化:

  2002 ─── Windows Forms (.NET Framework 1.0)
            │  GDI+ ベースの UI
            │  イベント駆動プログラミング
            │  ドラッグ&ドロップ設計
            │
  2006 ─── WPF (.NET Framework 3.0)
            │  XAML + データバインディング
            │  DirectX ベースのベクターレンダリング
            │  デスクトップアプリの新標準
            │
  2012 ─── WinRT / Windows 8 アプリ
            │  サンドボックス、タッチ対応
            │  Modern UI (Metro)
            │
  2015 ─── UWP (Universal Windows Platform)
            │  Windows 10 統一プラットフォーム
            │  Fluent Design System
            │  Microsoft Store 配布
            │  XAML Islands（WPF からの段階移行）
            │
  2021 ─── WinUI 3 (Windows App SDK)
            │  UWP の UI を Win32 アプリで使用可能
            │  最新の Fluent Design
            │  .NET 8+ / C++ 対応
            │  Win32 API フルアクセス
            │
  2022 ─── .NET MAUI
               クロスプラットフォーム
               Windows + macOS + iOS + Android
               Xamarin.Forms の後継
```

### 1.2 詳細な比較表

```
フレームワーク比較:

  項目       │ WPF        │ WinUI 3    │ UWP        │ MAUI
  ──────────┼───────────┼───────────┼───────────┼──────────
  対応 OS    │ Windows    │ Windows    │ Windows    │ マルチ
  .NET      │ Framework/8│ 8+         │ 制限あり   │ 8+
  UI 技術   │ XAML       │ XAML       │ XAML       │ XAML
  デザイン   │ 従来       │ Fluent     │ Fluent     │ ネイティブ
  Win32 API │ 完全       │ 完全       │ 制限       │ 制限
  配布      │ 自由       │ 自由       │ Store 推奨 │ 自由
  状態      │ 保守       │ 推奨       │ 非推奨     │ 活発

  選定ガイド:
    新規 Windows アプリ → WinUI 3
    既存 WPF 資産あり → WPF 継続（段階的 WinUI 3 移行）
    クロスプラットフォーム → MAUI
    UWP アプリ → WinUI 3 へ移行推奨
```

### 1.3 レンダリングエンジンの違い

```
レンダリングアーキテクチャ:

  Windows Forms:
    GDI/GDI+ → CPU レンダリング → ラスタライズ
    ・ピクセルベース描画
    ・高DPIでぼやける場合あり
    ・シンプルだが表現力に限界

  WPF:
    XAML → MIL (Media Integration Layer) → DirectX 9/11 → GPU レンダリング
    ・ベクターベース描画
    ・高DPI対応
    ・3D、アニメーション、エフェクト

  WinUI 3:
    XAML → Windows.UI.Composition → DirectX 12 → GPU レンダリング
    ・コンポジションベース描画
    ・Mica/Acrylic エフェクト
    ・最高のパフォーマンス
    ・DirectComposition による滑らかなアニメーション

  MAUI:
    XAML → ハンドラー → 各プラットフォームネイティブ UI
    ・Windows では WinUI 3 にマッピング
    ・macOS では Catalyst
    ・抽象化レイヤーによるオーバーヘッドあり
```

---

## 2. XAML の基礎

### 2.1 XAML の基本構文

```xml
<!-- XAML の基本構文 -->
<Window
    x:Class="MyApp.MainWindow"
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    Title="My Application" Height="600" Width="800">

    <Grid>
        <!-- 行と列の定義 -->
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto" />
            <RowDefinition Height="*" />
            <RowDefinition Height="Auto" />
        </Grid.RowDefinitions>

        <!-- ヘッダー -->
        <TextBlock Grid.Row="0" Text="ヘッダー"
                   FontSize="24" Margin="16" />

        <!-- コンテンツ -->
        <StackPanel Grid.Row="1" Margin="16">
            <TextBox x:Name="NameInput"
                     PlaceholderText="名前を入力"
                     Text="{x:Bind ViewModel.Name, Mode=TwoWay}" />

            <Button Content="挨拶"
                    Click="OnGreetClick"
                    Margin="0,8,0,0" />

            <TextBlock Text="{x:Bind ViewModel.Greeting, Mode=OneWay}"
                       Margin="0,16,0,0" />
        </StackPanel>

        <!-- フッター -->
        <TextBlock Grid.Row="2" Text="(C) 2024" Margin="16" />
    </Grid>
</Window>
```

### 2.2 名前空間と xmlns 宣言

```xml
<!-- 名前空間の詳細 -->
<Page
    <!-- デフォルトの XAML 名前空間（UI コントロール） -->
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"

    <!-- x: 名前空間（XAML 組み込み機能: x:Name, x:Class, x:Bind） -->
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"

    <!-- ローカルの名前空間（自作クラスの参照） -->
    xmlns:local="using:MyApp"

    <!-- ViewModel 名前空間 -->
    xmlns:vm="using:MyApp.ViewModels"

    <!-- CommunityToolkit のコントロール -->
    xmlns:toolkit="using:CommunityToolkit.WinUI.UI.Controls"

    <!-- デザイン時データ（実行時は無視される） -->
    xmlns:d="http://schemas.microsoft.com/expression/blend/2008"
    xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
    mc:Ignorable="d"
    d:DesignHeight="600"
    d:DesignWidth="800">

    <!-- ページ内容 -->
</Page>
```

### 2.3 レイアウトパネルの詳細

```xml
<!-- StackPanel: 水平または垂直の直列配置 -->
<StackPanel Orientation="Vertical" Spacing="8">
    <TextBlock Text="項目1" />
    <TextBlock Text="項目2" />
    <TextBlock Text="項目3" />
</StackPanel>

<!-- Grid: 行と列のマトリックス配置 -->
<Grid ColumnSpacing="16" RowSpacing="8" Padding="24">
    <Grid.ColumnDefinitions>
        <ColumnDefinition Width="Auto" />   <!-- コンテンツに合わせる -->
        <ColumnDefinition Width="*" />      <!-- 残りのスペースを占有 -->
        <ColumnDefinition Width="2*" />     <!-- 残りの2倍 -->
        <ColumnDefinition Width="200" />    <!-- 固定幅 -->
    </Grid.ColumnDefinitions>
    <Grid.RowDefinitions>
        <RowDefinition Height="Auto" />
        <RowDefinition Height="*" />
    </Grid.RowDefinitions>

    <TextBlock Grid.Row="0" Grid.Column="0" Text="左上" />
    <TextBlock Grid.Row="0" Grid.Column="1" Grid.ColumnSpan="2" Text="2列にまたがる" />
    <TextBlock Grid.Row="1" Grid.Column="0" Grid.RowSpan="1" Text="左下" />
</Grid>

<!-- RelativePanel: 相対位置指定 -->
<RelativePanel>
    <TextBlock x:Name="Header" Text="ヘッダー"
               RelativePanel.AlignTopWithPanel="True"
               RelativePanel.AlignLeftWithPanel="True"
               RelativePanel.AlignRightWithPanel="True" />

    <TextBox x:Name="SearchBox"
             RelativePanel.Below="Header"
             RelativePanel.AlignLeftWithPanel="True"
             Width="300" Margin="0,8,0,0" />

    <Button Content="検索"
            RelativePanel.Below="Header"
            RelativePanel.RightOf="SearchBox"
            Margin="8,8,0,0" />
</RelativePanel>

<!-- Canvas: 絶対座標指定 -->
<Canvas Width="400" Height="300">
    <Rectangle Canvas.Left="10" Canvas.Top="10"
               Width="100" Height="80"
               Fill="Blue" />
    <Ellipse Canvas.Left="150" Canvas.Top="50"
             Width="80" Height="80"
             Fill="Red" />
</Canvas>
```

### 2.4 リソースとスタイル

```xml
<!-- リソースの定義と使用 -->
<Page.Resources>
    <!-- 色の定義 -->
    <Color x:Key="PrimaryColor">#6366F1</Color>
    <SolidColorBrush x:Key="PrimaryBrush" Color="{StaticResource PrimaryColor}" />

    <!-- マージンの共通定義 -->
    <Thickness x:Key="StandardMargin">16,8,16,8</Thickness>

    <!-- 文字列リソース -->
    <x:String x:Key="AppTitle">マイアプリケーション</x:String>

    <!-- ボタンのスタイル定義 -->
    <Style x:Key="PrimaryButtonStyle" TargetType="Button">
        <Setter Property="Background" Value="{StaticResource PrimaryBrush}" />
        <Setter Property="Foreground" Value="White" />
        <Setter Property="CornerRadius" Value="8" />
        <Setter Property="Padding" Value="24,12" />
        <Setter Property="FontWeight" Value="SemiBold" />
    </Style>

    <!-- 暗黙的スタイル（x:Key なし → 全ての TextBlock に適用） -->
    <Style TargetType="TextBlock">
        <Setter Property="FontFamily" Value="Segoe UI" />
        <Setter Property="FontSize" Value="14" />
    </Style>

    <!-- スタイルの継承 -->
    <Style x:Key="DangerButtonStyle" TargetType="Button"
           BasedOn="{StaticResource PrimaryButtonStyle}">
        <Setter Property="Background" Value="#EF4444" />
    </Style>
</Page.Resources>

<!-- リソースの使用 -->
<StackPanel>
    <TextBlock Text="{StaticResource AppTitle}" />
    <Button Style="{StaticResource PrimaryButtonStyle}" Content="保存" />
    <Button Style="{StaticResource DangerButtonStyle}" Content="削除" />
</StackPanel>
```

---

## 3. データバインディング

### 3.1 バインディングモードの詳細

```
バインディングモード:

  OneWay:     ViewModel → View（表示のみ）
              プロパティ変更 → UI 自動更新
              例: テキスト表示、リスト表示

  TwoWay:     ViewModel <-> View（双方向）
              UI 入力 → プロパティ自動更新
              例: テキスト入力、チェックボックス、スライダー

  OneTime:    初期値のみ設定（変更追跡なし）
              パフォーマンス最適
              例: 定数表示、設定値の初期読み込み

  OneWayToSource: View → ViewModel（逆方向のみ）
              UI の値を ViewModel に反映するが、逆は行わない
              例: パスワード入力の値取得
```

### 3.2 x:Bind と Binding の違い

```xml
<!-- x:Bind（コンパイル時バインディング） — WinUI 3 推奨 -->
<!-- 利点: 型安全、コンパイル時エラー検出、高速 -->
<TextBlock Text="{x:Bind ViewModel.Name, Mode=OneWay}" />
<TextBox Text="{x:Bind ViewModel.Email, Mode=TwoWay, UpdateSourceTrigger=PropertyChanged}" />
<Button Command="{x:Bind ViewModel.SaveCommand}" />

<!-- Binding（ランタイムバインディング） — WPF 互換 -->
<!-- 利点: DataContext を経由した柔軟なバインディング -->
<TextBlock Text="{Binding Name}" />
<TextBox Text="{Binding Email, Mode=TwoWay, UpdateSourceTrigger=PropertyChanged}" />
<Button Command="{Binding SaveCommand}" />

<!--
x:Bind vs Binding の比較:
┌─────────────┬────────────────────┬────────────────────┐
│ 項目        │ x:Bind             │ Binding            │
├─────────────┼────────────────────┼────────────────────┤
│ 解決時期    │ コンパイル時       │ ランタイム         │
│ 型安全      │ あり               │ なし               │
│ パフォーマンス│ 高速             │ やや遅い           │
│ デフォルトモード│ OneTime         │ OneWay             │
│ DataContext │ 不要（コード直接）│ 必要               │
│ 対応FW      │ WinUI 3 / UWP    │ WPF / WinUI 3     │
│ 式の記法    │ 関数呼び出し可   │ コンバーター必要  │
└─────────────┴────────────────────┴────────────────────┘
-->
```

### 3.3 ViewModel の基本実装（INotifyPropertyChanged）

```csharp
// ViewModel — INotifyPropertyChanged 実装（手動）
public class MainViewModel : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;

    private string _name = "";
    public string Name
    {
        get => _name;
        set
        {
            if (_name != value)
            {
                _name = value;
                OnPropertyChanged(nameof(Name));
                OnPropertyChanged(nameof(Greeting));
            }
        }
    }

    private string _email = "";
    public string Email
    {
        get => _email;
        set
        {
            if (_email != value)
            {
                _email = value;
                OnPropertyChanged(nameof(Email));
                OnPropertyChanged(nameof(IsValid));
            }
        }
    }

    public string Greeting => string.IsNullOrEmpty(Name)
        ? ""
        : $"Hello, {Name}!";

    public bool IsValid => !string.IsNullOrEmpty(Name) &&
                           !string.IsNullOrEmpty(Email) &&
                           Email.Contains('@');

    protected void OnPropertyChanged(string propertyName)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
```

### 3.4 コレクションバインディング

```csharp
// ObservableCollection を使ったリストバインディング
using System.Collections.ObjectModel;

public class TaskListViewModel : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;

    // ObservableCollection: 追加・削除が自動で UI に反映される
    public ObservableCollection<TaskItem> Tasks { get; } = new();

    private TaskItem? _selectedTask;
    public TaskItem? SelectedTask
    {
        get => _selectedTask;
        set
        {
            _selectedTask = value;
            OnPropertyChanged(nameof(SelectedTask));
            OnPropertyChanged(nameof(HasSelection));
        }
    }

    public bool HasSelection => SelectedTask != null;

    public void AddTask(string title)
    {
        Tasks.Add(new TaskItem { Title = title, CreatedAt = DateTime.Now });
    }

    public void RemoveTask(TaskItem task)
    {
        Tasks.Remove(task);
        if (SelectedTask == task) SelectedTask = null;
    }

    protected void OnPropertyChanged(string propertyName)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}

public class TaskItem : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;

    private string _title = "";
    public string Title
    {
        get => _title;
        set { _title = value; PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Title))); }
    }

    private bool _isCompleted;
    public bool IsCompleted
    {
        get => _isCompleted;
        set { _isCompleted = value; PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(IsCompleted))); }
    }

    public DateTime CreatedAt { get; set; }
}
```

```xml
<!-- リストのバインディング -->
<ListView ItemsSource="{x:Bind ViewModel.Tasks}"
          SelectedItem="{x:Bind ViewModel.SelectedTask, Mode=TwoWay}">
    <ListView.ItemTemplate>
        <DataTemplate x:DataType="local:TaskItem">
            <StackPanel Orientation="Horizontal" Spacing="8" Padding="8">
                <CheckBox IsChecked="{x:Bind IsCompleted, Mode=TwoWay}" />
                <TextBlock Text="{x:Bind Title}" VerticalAlignment="Center" />
                <TextBlock Text="{x:Bind CreatedAt}" Opacity="0.6"
                           VerticalAlignment="Center" FontSize="12" />
            </StackPanel>
        </DataTemplate>
    </ListView.ItemTemplate>
</ListView>

<!-- 選択状態に応じた UI 表示制御 -->
<Button Content="削除"
        IsEnabled="{x:Bind ViewModel.HasSelection, Mode=OneWay}"
        Click="OnDeleteClick" />
```

### 3.5 値コンバーター

```csharp
// 値コンバーター: bool → Visibility 変換
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Data;

public class BoolToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, string language)
    {
        if (value is bool boolValue)
        {
            // parameter が "Invert" なら反転
            bool invert = parameter?.ToString() == "Invert";
            bool isVisible = invert ? !boolValue : boolValue;
            return isVisible ? Visibility.Visible : Visibility.Collapsed;
        }
        return Visibility.Collapsed;
    }

    public object ConvertBack(object value, Type targetType, object parameter, string language)
    {
        if (value is Visibility visibility)
        {
            bool invert = parameter?.ToString() == "Invert";
            bool result = visibility == Visibility.Visible;
            return invert ? !result : result;
        }
        return false;
    }
}

// 日付フォーマットコンバーター
public class DateFormatConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, string language)
    {
        if (value is DateTime dateTime)
        {
            string format = parameter?.ToString() ?? "yyyy/MM/dd HH:mm";
            return dateTime.ToString(format);
        }
        return value?.ToString() ?? "";
    }

    public object ConvertBack(object value, Type targetType, object parameter, string language)
    {
        if (value is string str && DateTime.TryParse(str, out var result))
        {
            return result;
        }
        return DateTime.MinValue;
    }
}

// 数値 → 色の変換（例: 優先度に応じた色）
public class PriorityToColorConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, string language)
    {
        if (value is string priority)
        {
            return priority switch
            {
                "high" => new SolidColorBrush(Windows.UI.Color.FromArgb(255, 239, 68, 68)),   // 赤
                "medium" => new SolidColorBrush(Windows.UI.Color.FromArgb(255, 245, 158, 11)), // 黄
                "low" => new SolidColorBrush(Windows.UI.Color.FromArgb(255, 34, 197, 94)),     // 緑
                _ => new SolidColorBrush(Windows.UI.Color.FromArgb(255, 156, 163, 175)),       // グレー
            };
        }
        return new SolidColorBrush(Windows.UI.Color.FromArgb(255, 156, 163, 175));
    }

    public object ConvertBack(object value, Type targetType, object parameter, string language)
    {
        throw new NotImplementedException();
    }
}
```

```xml
<!-- コンバーターの使用 -->
<Page.Resources>
    <local:BoolToVisibilityConverter x:Key="BoolToVisibility" />
    <local:DateFormatConverter x:Key="DateFormat" />
    <local:PriorityToColorConverter x:Key="PriorityColor" />
</Page.Resources>

<!-- Visibility の制御 -->
<ProgressRing Visibility="{x:Bind ViewModel.IsLoading, Mode=OneWay,
              Converter={StaticResource BoolToVisibility}}" />

<!-- 非表示にする場合は parameter に Invert を指定 -->
<TextBlock Text="データなし"
           Visibility="{x:Bind ViewModel.HasData, Mode=OneWay,
           Converter={StaticResource BoolToVisibility}, ConverterParameter=Invert}" />

<!-- 日付のフォーマット -->
<TextBlock Text="{x:Bind ViewModel.CreatedAt, Mode=OneWay,
           Converter={StaticResource DateFormat}, ConverterParameter='yyyy年MM月dd日'}" />

<!-- 優先度に応じた色表示 -->
<Border Background="{x:Bind Priority, Converter={StaticResource PriorityColor}}"
        CornerRadius="4" Padding="8,4">
    <TextBlock Text="{x:Bind Priority}" Foreground="White" />
</Border>
```

---

## 4. MVVM パターン

### 4.1 MVVM のアーキテクチャ

```
MVVM（Model-View-ViewModel）:

  ┌──────────┐
  │   View   │  XAML + コードビハインド
  │  (XAML)  │  UI の表示とユーザー入力
  └────┬─────┘
       │ データバインディング
       │ コマンドバインディング
  ┌────▼─────┐
  │ViewModel │  プレゼンテーションロジック
  │          │  INotifyPropertyChanged
  │          │  ICommand
  └────┬─────┘
       │ 依存性注入
  ┌────▼─────┐
  │  Model   │  ビジネスロジック
  │          │  データアクセス
  └──────────┘

  利点:
    - UI とロジックの分離
    - テスタビリティ向上
    - デザイナーと開発者の分業
    - 再利用性の向上
```

### 4.2 CommunityToolkit.Mvvm を使った簡潔な ViewModel

```csharp
// CommunityToolkit.Mvvm を使った簡潔な ViewModel
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

public partial class MainViewModel : ObservableObject
{
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Greeting))]
    private string _name = "";

    public string Greeting => string.IsNullOrEmpty(Name)
        ? "" : $"Hello, {Name}!";

    [RelayCommand]
    private async Task SaveAsync()
    {
        await _fileService.SaveAsync(Name);
    }
}
```

### 4.3 依存性注入との組み合わせ

```csharp
// サービスの定義
public interface ITaskService
{
    Task<IReadOnlyList<TaskItem>> GetAllAsync();
    Task<TaskItem> CreateAsync(string title);
    Task UpdateAsync(TaskItem task);
    Task DeleteAsync(int id);
}

// サービスの実装
public class TaskService : ITaskService
{
    private readonly HttpClient _httpClient;

    public TaskService(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    public async Task<IReadOnlyList<TaskItem>> GetAllAsync()
    {
        var response = await _httpClient.GetAsync("/api/tasks");
        response.EnsureSuccessStatusCode();
        var tasks = await response.Content.ReadFromJsonAsync<List<TaskItem>>();
        return tasks ?? new List<TaskItem>();
    }

    public async Task<TaskItem> CreateAsync(string title)
    {
        var response = await _httpClient.PostAsJsonAsync("/api/tasks", new { title });
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<TaskItem>()
            ?? throw new InvalidOperationException("タスクの作成に失敗しました");
    }

    public async Task UpdateAsync(TaskItem task)
    {
        var response = await _httpClient.PutAsJsonAsync($"/api/tasks/{task.Id}", task);
        response.EnsureSuccessStatusCode();
    }

    public async Task DeleteAsync(int id)
    {
        var response = await _httpClient.DeleteAsync($"/api/tasks/{id}");
        response.EnsureSuccessStatusCode();
    }
}

// ViewModel: サービスを依存性注入で受け取る
public partial class TaskListViewModel : ObservableObject
{
    private readonly ITaskService _taskService;

    public TaskListViewModel(ITaskService taskService)
    {
        _taskService = taskService;
    }

    [ObservableProperty]
    private ObservableCollection<TaskItem> _tasks = new();

    [ObservableProperty]
    private bool _isLoading;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(DeleteTaskCommand))]
    private TaskItem? _selectedTask;

    [ObservableProperty]
    private string _newTaskTitle = "";

    [RelayCommand]
    private async Task LoadTasksAsync()
    {
        try
        {
            IsLoading = true;
            var tasks = await _taskService.GetAllAsync();
            Tasks = new ObservableCollection<TaskItem>(tasks);
        }
        catch (Exception ex)
        {
            // エラーハンドリング
            Debug.WriteLine($"タスク読み込みエラー: {ex.Message}");
        }
        finally
        {
            IsLoading = false;
        }
    }

    [RelayCommand]
    private async Task AddTaskAsync()
    {
        if (string.IsNullOrWhiteSpace(NewTaskTitle)) return;

        var task = await _taskService.CreateAsync(NewTaskTitle);
        Tasks.Add(task);
        NewTaskTitle = "";
    }

    private bool CanDeleteTask() => SelectedTask != null;

    [RelayCommand(CanExecute = nameof(CanDeleteTask))]
    private async Task DeleteTaskAsync()
    {
        if (SelectedTask == null) return;

        await _taskService.DeleteAsync(SelectedTask.Id);
        Tasks.Remove(SelectedTask);
        SelectedTask = null;
    }
}
```

### 4.4 DI コンテナの設定（WinUI 3）

```csharp
// App.xaml.cs — DI コンテナの設定
using Microsoft.Extensions.DependencyInjection;
using Microsoft.UI.Xaml;

public partial class App : Application
{
    public IServiceProvider Services { get; }
    public static new App Current => (App)Application.Current;

    public App()
    {
        this.InitializeComponent();

        // DI コンテナの構築
        var services = new ServiceCollection();

        // サービスの登録
        services.AddHttpClient<ITaskService, TaskService>(client =>
        {
            client.BaseAddress = new Uri("https://api.example.com");
        });

        // ViewModel の登録
        services.AddTransient<MainViewModel>();
        services.AddTransient<TaskListViewModel>();
        services.AddTransient<SettingsViewModel>();

        // ナビゲーションサービスの登録
        services.AddSingleton<INavigationService, NavigationService>();

        Services = services.BuildServiceProvider();
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        var window = new MainWindow();
        window.Activate();
    }
}

// ページでの ViewModel 取得
public sealed partial class TaskListPage : Page
{
    public TaskListViewModel ViewModel { get; }

    public TaskListPage()
    {
        ViewModel = App.Current.Services.GetRequiredService<TaskListViewModel>();
        this.InitializeComponent();

        // ページ読み込み時にデータを取得
        Loaded += async (_, _) => await ViewModel.LoadTasksCommand.ExecuteAsync(null);
    }
}
```

---

## 5. 各フレームワークのコード比較

### 5.1 WPF での実装例

```csharp
// WPF: MainWindow.xaml.cs
using System.Windows;

namespace WpfApp;

public partial class MainWindow : Window
{
    public MainViewModel ViewModel { get; }

    public MainWindow()
    {
        ViewModel = new MainViewModel();
        DataContext = ViewModel;
        InitializeComponent();
    }
}
```

```xml
<!-- WPF: MainWindow.xaml -->
<Window x:Class="WpfApp.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="WPF App" Height="400" Width="600">
    <StackPanel Margin="16">
        <!-- WPF では Binding（ランタイム）を使用 -->
        <TextBox Text="{Binding Name, UpdateSourceTrigger=PropertyChanged}"
                 Margin="0,0,0,8" />
        <Button Content="挨拶" Command="{Binding GreetCommand}"
                Margin="0,0,0,8" />
        <TextBlock Text="{Binding Greeting}" FontSize="18" />
    </StackPanel>
</Window>
```

### 5.2 WinUI 3 での実装例

```csharp
// WinUI 3: MainWindow.xaml.cs
using Microsoft.UI.Xaml;

namespace WinUIApp;

public sealed partial class MainWindow : Window
{
    public MainViewModel ViewModel { get; } = new();

    public MainWindow()
    {
        this.InitializeComponent();
    }
}
```

```xml
<!-- WinUI 3: MainWindow.xaml -->
<Window x:Class="WinUIApp.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="WinUI 3 App">
    <StackPanel Margin="16" Spacing="8">
        <!-- WinUI 3 では x:Bind（コンパイル時）を推奨 -->
        <TextBox Text="{x:Bind ViewModel.Name, Mode=TwoWay, UpdateSourceTrigger=PropertyChanged}" />
        <Button Content="挨拶" Command="{x:Bind ViewModel.GreetCommand}" />
        <TextBlock Text="{x:Bind ViewModel.Greeting, Mode=OneWay}"
                   Style="{StaticResource SubtitleTextBlockStyle}" />
    </StackPanel>
</Window>
```

### 5.3 .NET MAUI での実装例

```csharp
// MAUI: MainPage.xaml.cs
namespace MauiApp;

public partial class MainPage : ContentPage
{
    public MainViewModel ViewModel { get; } = new();

    public MainPage()
    {
        BindingContext = ViewModel;
        InitializeComponent();
    }
}
```

```xml
<!-- MAUI: MainPage.xaml -->
<ContentPage xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
             xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml"
             x:Class="MauiApp.MainPage">
    <VerticalStackLayout Spacing="8" Padding="16">
        <!-- MAUI では Binding を使用 -->
        <Entry Text="{Binding Name}" Placeholder="名前を入力" />
        <Button Text="挨拶" Command="{Binding GreetCommand}" />
        <Label Text="{Binding Greeting}" FontSize="18" />
    </VerticalStackLayout>
</ContentPage>
```

### 5.4 フレームワーク間の API 対応表

```
主要コントロールの名前の違い:

  概念         │ WPF           │ WinUI 3       │ MAUI
  ─────────────┼──────────────┼──────────────┼──────────────
  テキスト表示 │ TextBlock     │ TextBlock     │ Label
  テキスト入力 │ TextBox       │ TextBox       │ Entry
  複数行入力   │ TextBox       │ TextBox       │ Editor
  ボタン       │ Button        │ Button        │ Button
  チェック     │ CheckBox      │ CheckBox      │ CheckBox
  ドロップダウン│ ComboBox      │ ComboBox      │ Picker
  リスト       │ ListView      │ ListView      │ CollectionView
  スクロール   │ ScrollViewer  │ ScrollViewer  │ ScrollView
  垂直並べ     │ StackPanel    │ StackPanel    │ VerticalStackLayout
  水平並べ     │ StackPanel    │ StackPanel    │ HorizontalStackLayout
  グリッド     │ Grid          │ Grid          │ Grid
  画像         │ Image         │ Image         │ Image
  スライダー   │ Slider        │ Slider        │ Slider
  トグル       │ ToggleButton  │ ToggleSwitch  │ Switch
  ナビゲーション│ Frame         │ NavigationView│ Shell
  ダイアログ   │ MessageBox    │ ContentDialog │ DisplayAlert
```

---

## 6. Windows Forms からの移行ガイド

### 6.1 移行パスの選択

```
Windows Forms → WPF / WinUI 3 移行の判断基準:

  WForms → WPF:
    ・.NET Framework からの段階的移行
    ・既存の WinForms コントロールを WPF にホスティング可能
    ・WindowsFormsHost コントロールで共存

  WForms → WinUI 3:
    ・モダンな UI が必要
    ・新しい .NET（.NET 8+）に移行済み
    ・Fluent Design が必要

  移行の優先順位:
    1. ビジネスロジックの分離（UI から独立させる）
    2. イベントハンドラ → MVVM パターンへの変換
    3. UI の段階的な書き換え
    4. テストの追加
```

### 6.2 イベント駆動から MVVM への変換

```csharp
// Before: Windows Forms のイベント駆動スタイル
// ボタンクリック → 直接 DB 操作 → UI 更新
public partial class OrderForm : Form
{
    private void btnSubmit_Click(object sender, EventArgs e)
    {
        // ビジネスロジックが UI に密結合
        var order = new Order
        {
            CustomerName = txtCustomerName.Text,
            Amount = decimal.Parse(txtAmount.Text),
        };

        using var conn = new SqlConnection(connectionString);
        conn.Open();
        // ... SQL 実行

        lblStatus.Text = "注文が完了しました";
        txtCustomerName.Text = "";
        txtAmount.Text = "";
    }
}
```

```csharp
// After: MVVM パターン（WPF / WinUI 3）
// ViewModel: UI から完全に独立したロジック
public partial class OrderViewModel : ObservableObject
{
    private readonly IOrderService _orderService;

    public OrderViewModel(IOrderService orderService)
    {
        _orderService = orderService;
    }

    [ObservableProperty]
    private string _customerName = "";

    [ObservableProperty]
    private string _amount = "";

    [ObservableProperty]
    private string _statusMessage = "";

    [RelayCommand]
    private async Task SubmitOrderAsync()
    {
        if (!decimal.TryParse(Amount, out var amountValue))
        {
            StatusMessage = "金額が不正です";
            return;
        }

        var order = new Order
        {
            CustomerName = CustomerName,
            Amount = amountValue,
        };

        await _orderService.CreateAsync(order);
        StatusMessage = "注文が完了しました";
        CustomerName = "";
        Amount = "";
    }
}
```

---

## 7. パフォーマンス比較

```
フレームワーク別パフォーマンス特性:

  ┌────────────┬──────────┬──────────┬──────────┬──────────┐
  │ 指標       │ WForms   │ WPF      │ WinUI 3  │ MAUI     │
  ├────────────┼──────────┼──────────┼──────────┼──────────┤
  │ 起動時間   │ 最速     │ やや遅い │ やや遅い │ 遅い     │
  │ メモリ使用 │ 最小     │ 中       │ 中       │ やや多い │
  │ GPU活用    │ なし     │ あり     │ 最適     │ OS依存   │
  │ 大量データ │ 良好     │ 仮想化   │ 仮想化   │ 仮想化   │
  │ アニメーション│ 限定的 │ 良好     │ 最良     │ OS依存   │
  │ 高DPI対応  │ 要設定   │ 良好     │ 最良     │ 自動     │
  └────────────┴──────────┴──────────┴──────────┴──────────┘

  パフォーマンス最適化のポイント:

  WPF:
    ・VirtualizingStackPanel で大量リストの仮想化
    ・Freezable オブジェクトの凍結（ブラシ、ジオメトリ等）
    ・BindingMode=OneTime の積極活用
    ・非同期データ読み込み（async/await）

  WinUI 3:
    ・x:Bind によるコンパイル時バインディング
    ・x:Load / x:DeferLoadStrategy で遅延読み込み
    ・ListView の ItemsRepeater への置き換え（大量データ時）
    ・Composition API によるアニメーション最適化
```

---

## 8. アクセシビリティ対応

```csharp
// WinUI 3 でのアクセシビリティ実装例
// AutomationProperties で支援技術に情報を提供
```

```xml
<!-- アクセシビリティ対応の XAML -->
<StackPanel>
    <!-- スクリーンリーダー向けの名前設定 -->
    <TextBox
        AutomationProperties.Name="ユーザー名入力欄"
        AutomationProperties.HelpText="ログインに使用するユーザー名を入力してください"
        PlaceholderText="ユーザー名" />

    <!-- ランドマークの設定 -->
    <NavigationView
        AutomationProperties.LandmarkType="Navigation"
        AutomationProperties.Name="メインナビゲーション">
        <!-- ... -->
    </NavigationView>

    <!-- ライブリージョン（動的に変化するテキスト） -->
    <TextBlock
        x:Name="StatusText"
        AutomationProperties.LiveSetting="Polite"
        AutomationProperties.Name="ステータスメッセージ" />

    <!-- 高コントラストモード対応 -->
    <Button Content="保存"
            Style="{ThemeResource AccentButtonStyle}">
        <!-- ThemeResource を使用すれば高コントラストモードに自動対応 -->
    </Button>

    <!-- キーボードナビゲーション -->
    <Grid KeyboardAcceleratorPlacementMode="Hidden">
        <Grid.KeyboardAccelerators>
            <KeyboardAccelerator Key="S" Modifiers="Control"
                                 Invoked="SaveAccelerator_Invoked" />
        </Grid.KeyboardAccelerators>
    </Grid>
</StackPanel>
```

```csharp
// タブオーダーの制御
public sealed partial class LoginPage : Page
{
    public LoginPage()
    {
        InitializeComponent();

        // タブオーダーの明示的な設定
        UsernameBox.TabIndex = 1;
        PasswordBox.TabIndex = 2;
        LoginButton.TabIndex = 3;
        ForgotPasswordLink.TabIndex = 4;

        // フォーカスの初期設定
        Loaded += (_, _) => UsernameBox.Focus(FocusState.Programmatic);
    }
}
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---

## FAQ

### Q1: WPF から WinUI 3 に移行すべきか？
新規開発なら WinUI 3 推奨。既存 WPF アプリは安定稼働中なら急ぐ必要なし。XAML Islands で段階的に WinUI 3 コントロールを導入可能。ただし、WPF は .NET 8 以降も引き続きサポートされるため、移行の緊急性は低い。WinUI 3 への移行が特に有効なのは、Fluent Design への対応が必要な場合や、Windows 11 固有の機能（Mica、Snap Layout など）を活用したい場合である。

### Q2: MAUI は実用段階か？
Windows + macOS では実用可能。iOS/Android も対応するがネイティブの洗練さには劣る。Windows 専用なら WinUI 3 の方が良い。MAUI は Xamarin.Forms の後継であり、クロスプラットフォーム展開が必要な業務アプリケーションには適している。ただし、プラットフォーム固有の高度な UI カスタマイズが必要な場合は、各 OS のネイティブフレームワークを直接使用する方が効率的である。

### Q3: CommunityToolkit.Mvvm を使うべきか？
推奨。INotifyPropertyChanged のボイラープレートを Source Generator で自動生成。RelayCommand も簡潔に書ける。手動で実装した場合と比較して、コード量を50-70%削減できる。WPF、WinUI 3、MAUI のいずれでも使用可能であり、フレームワーク間での ViewModel の共有も容易になる。NuGet パッケージとして `CommunityToolkit.Mvvm` をインストールするだけで利用できる。

### Q4: Windows Forms はまだ使えるか？
使える。.NET 8 以降でもサポートが継続されており、新機能も追加されている。既存の Windows Forms アプリケーションを急いで移行する必要はない。ただし、新規開発で Windows Forms を選択する積極的な理由は少ない。高DPI 対応やモダンな UI デザインが必要な場合は、WPF または WinUI 3 を選択すべきである。

### Q5: WinUI 3 でまだサポートされていない WPF の機能は？
WinUI 3 には WPF の一部機能がまだ移植されていない。主なものとして、FlowDocument（リッチテキスト表示）、XPS 印刷サポート、一部の 3D レンダリング機能、RibbonControl などがある。これらの機能が必要な場合は WPF を継続使用するか、サードパーティライブラリで代替する必要がある。

---

## まとめ

| フレームワーク | 推奨用途 | 状態 |
|-------------|---------|------|
| WinUI 3 | Windows ネイティブ新規開発 | 推奨 |
| WPF | 既存アプリの保守・拡張 | 保守モード |
| MAUI | クロスプラットフォーム | 活発 |
| UWP | なし（WinUI 3 へ移行推奨） | 非推奨 |
| Windows Forms | レガシーアプリの保守 | サポート継続 |

---

## 次に読むべきガイド

---

## 参考文献
1. Microsoft. "WinUI 3." learn.microsoft.com/windows/apps/winui, 2024.
2. Microsoft. "WPF Documentation." learn.microsoft.com/dotnet/desktop/wpf, 2024.
3. Microsoft. ".NET MAUI." learn.microsoft.com/dotnet/maui, 2024.
4. Microsoft. "CommunityToolkit.Mvvm." learn.microsoft.com/dotnet/communitytoolkit/mvvm, 2024.
5. Microsoft. "Windows Forms." learn.microsoft.com/dotnet/desktop/winforms, 2024.
6. Microsoft. "XAML Overview." learn.microsoft.com/windows/uwp/xaml-platform, 2024.
7. Microsoft. "Windows App SDK." learn.microsoft.com/windows/apps/windows-app-sdk, 2024.

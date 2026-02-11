# Windows UI フレームワーク比較

> Windows ネイティブ UI フレームワークは WPF から WinUI 3 へ進化した。WPF、WinUI 3、UWP、MAUI の歴史的経緯と使い分け、XAML の基礎、データバインディング、MVVM パターンまで解説する。

## この章で学ぶこと

- [ ] Windows UI フレームワークの歴史と選定基準を理解する
- [ ] XAML の基礎構文を把握する
- [ ] MVVM パターンとデータバインディングを理解する

---

## 1. フレームワークの歴史と比較

```
Windows UI フレームワークの進化:

  2006 ─── WPF (.NET Framework)
            │  XAML + データバインディング
            │  デスクトップアプリの標準
            │
  2012 ─── WinRT / Windows 8 アプリ
            │  サンドボックス、タッチ対応
            │
  2015 ─── UWP (Universal Windows Platform)
            │  Windows 10 統一プラットフォーム
            │  Fluent Design System
            │  Microsoft Store 配布
            │
  2021 ─── WinUI 3 (Windows App SDK)
            │  UWP の UI を Win32 アプリで使用可能
            │  最新の Fluent Design
            │  .NET 8+ / C++ 対応
            │
  2022 ─── .NET MAUI
               クロスプラットフォーム
               Windows + macOS + iOS + Android
```

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

---

## 2. XAML の基礎

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
        <TextBlock Grid.Row="2" Text="© 2024" Margin="16" />
    </Grid>
</Window>
```

---

## 3. データバインディング

```
バインディングモード:

  OneWay:     ViewModel → View（表示のみ）
  TwoWay:     ViewModel ↔ View（双方向、入力フォーム）
  OneTime:    初期値のみ（パフォーマンス最適）
  OneWayToSource: View → ViewModel
```

```csharp
// ViewModel — INotifyPropertyChanged 実装
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

    public string Greeting => string.IsNullOrEmpty(Name)
        ? ""
        : $"Hello, {Name}!";

    protected void OnPropertyChanged(string propertyName)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
```

---

## 4. MVVM パターン

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
    ✓ UI とロジックの分離
    ✓ テスタビリティ向上
    ✓ デザイナーと開発者の分業
    ✓ 再利用性の向上
```

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

---

## FAQ

### Q1: WPF から WinUI 3 に移行すべきか？
新規開発なら WinUI 3 推奨。既存 WPF アプリは安定稼働中なら急ぐ必要なし。XAML Islands で段階的に WinUI 3 コントロールを導入可能。

### Q2: MAUI は実用段階か？
Windows + macOS では実用可能。iOS/Android も対応するがネイティブの洗練さには劣る。Windows 専用なら WinUI 3 の方が良い。

### Q3: CommunityToolkit.Mvvm を使うべきか？
推奨。INotifyPropertyChanged のボイラープレートを Source Generator で自動生成。RelayCommand も簡潔に書ける。

---

## まとめ

| フレームワーク | 推奨用途 | 状態 |
|-------------|---------|------|
| WinUI 3 | Windows ネイティブ新規開発 | 推奨 |
| WPF | 既存アプリの保守・拡張 | 保守モード |
| MAUI | クロスプラットフォーム | 活発 |
| UWP | なし（WinUI 3 へ移行推奨） | 非推奨 |

---

## 次に読むべきガイド
→ [[01-winui3-basics.md]] — WinUI 3 の基本

---

## 参考文献
1. Microsoft. "WinUI 3." learn.microsoft.com/windows/apps/winui, 2024.
2. Microsoft. "WPF Documentation." learn.microsoft.com/dotnet/desktop/wpf, 2024.
3. Microsoft. ".NET MAUI." learn.microsoft.com/dotnet/maui, 2024.

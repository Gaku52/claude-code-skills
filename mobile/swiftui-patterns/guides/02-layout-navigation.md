# Layout & Navigation - SwiftUIレイアウトとナビゲーション完全ガイド

## 📋 目次

1. [概要](#概要)
2. [基本レイアウト - Stacks](#基本レイアウト---stacks)
3. [Spacer と Divider](#spacer-と-divider)
4. [フレームとサイズ調整](#フレームとサイズ調整)
5. [Alignment と Padding](#alignment-と-padding)
6. [GeometryReader](#geometryreader)
7. [Custom Layout (iOS 16+)](#custom-layout-ios-16)
8. [LazyStacks と ScrollView](#lazystacks-と-scrollview)
9. [Grid レイアウト](#grid-レイアウト)
10. [NavigationStack (iOS 16+)](#navigationstack-ios-16)
11. [Modal Presentation](#modal-presentation)
12. [TabView](#tabview)
13. [SplitView](#splitview)
14. [アダプティブレイアウト](#アダプティブレイアウト)
15. [よくある問題と解決策](#よくある問題と解決策)

## 概要

SwiftUIのレイアウトシステムは、宣言的で柔軟なUI構築を可能にします。このガイドでは、基本的なStacksから高度なカスタムレイアウト、ナビゲーションパターンまでを網羅します。

### このガイドの対象者

- SwiftUI初学者〜中級者
- レスポンシブなUIを構築したい開発者
- 複雑なナビゲーションフローを実装する開発者

### 学べること

- 効率的なレイアウト構築
- パフォーマンスの高いスクロールビュー
- 型安全なナビゲーション実装
- アダプティブデザインの実践

## 基本レイアウト - Stacks

### VStack - 垂直配置

```swift
struct VStackExampleView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Title")
                .font(.title)
                .fontWeight(.bold)

            Text("Subtitle")
                .font(.headline)
                .foregroundColor(.secondary)

            Text("Body text goes here. This is a longer piece of text that demonstrates how VStack handles multiple lines.")
                .font(.body)

            Divider()

            Text("Footer")
                .font(.caption)
                .foregroundColor(.gray)
        }
        .padding()
    }
}
```

### HStack - 水平配置

```swift
struct HStackExampleView: View {
    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            // 左側のアイコン
            Image(systemName: "star.fill")
                .font(.title)
                .foregroundColor(.yellow)

            // 中央のコンテンツ
            VStack(alignment: .leading, spacing: 4) {
                Text("Feature Title")
                    .font(.headline)
                Text("Description")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            Spacer()

            // 右側のバッジ
            Text("New")
                .font(.caption)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.blue)
                .foregroundColor(.white)
                .cornerRadius(8)
        }
        .padding()
    }
}
```

### ZStack - 重ね配置

```swift
struct ZStackExampleView: View {
    var body: some View {
        ZStack {
            // 背景
            LinearGradient(
                colors: [.blue, .purple],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            // 中央のカード
            VStack(spacing: 20) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 60))
                    .foregroundColor(.white)

                Text("Success!")
                    .font(.title)
                    .fontWeight(.bold)
                    .foregroundColor(.white)

                Text("Your action was completed successfully.")
                    .font(.body)
                    .foregroundColor(.white.opacity(0.9))
                    .multilineTextAlignment(.center)
            }
            .padding(40)
            .background(.ultraThinMaterial)
            .cornerRadius(20)
            .shadow(radius: 20)

            // 右上のバッジ
            VStack {
                HStack {
                    Spacer()
                    Circle()
                        .fill(.red)
                        .frame(width: 20, height: 20)
                        .overlay {
                            Text("3")
                                .font(.caption2)
                                .foregroundColor(.white)
                        }
                        .padding()
                }
                Spacer()
            }
        }
    }
}
```

### Stackの組み合わせ

```swift
struct ComplexLayoutView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // ヘッダー
                HStack {
                    VStack(alignment: .leading) {
                        Text("Dashboard")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                        Text("Welcome back!")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Image(systemName: "bell.badge")
                        .font(.title2)
                }
                .padding()

                // メトリクスカード
                HStack(spacing: 16) {
                    MetricCard(title: "Total", value: "1,234", icon: "chart.bar.fill", color: .blue)
                    MetricCard(title: "Active", value: "856", icon: "flame.fill", color: .orange)
                }
                .padding(.horizontal)

                // リスト
                VStack(alignment: .leading, spacing: 12) {
                    Text("Recent Activity")
                        .font(.headline)
                        .padding(.horizontal)

                    ForEach(0..<5) { index in
                        ActivityRow(index: index)
                    }
                }
            }
        }
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                Spacer()
            }
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct ActivityRow: View {
    let index: Int

    var body: some View {
        HStack {
            Circle()
                .fill(.blue)
                .frame(width: 8, height: 8)
            Text("Activity \(index + 1)")
            Spacer()
            Text("2m ago")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal)
    }
}
```

## Spacer と Divider

### Spacerの活用

```swift
struct SpacerExampleView: View {
    var body: some View {
        VStack {
            // 固定スペース
            Text("Top")

            // フレキシブルスペース
            Spacer()

            Text("Middle")

            Spacer()

            Text("Bottom")

            // 最小スペース指定
            Spacer(minLength: 20)

            Text("Very Bottom")
        }
        .padding()
    }
}

struct SpacerPatternView: View {
    var body: some View {
        VStack(spacing: 20) {
            // パターン1: 左寄せ
            HStack {
                Text("Left")
                Spacer()
            }

            // パターン2: 右寄せ
            HStack {
                Spacer()
                Text("Right")
            }

            // パターン3: 中央揃え
            HStack {
                Spacer()
                Text("Center")
                Spacer()
            }

            // パターン4: 両端揃え
            HStack {
                Text("Left")
                Spacer()
                Text("Right")
            }

            // パターン5: 等間隔配置
            HStack {
                Text("A")
                Spacer()
                Text("B")
                Spacer()
                Text("C")
            }
        }
        .padding()
    }
}
```

### Dividerの使用

```swift
struct DividerExampleView: View {
    var body: some View {
        VStack {
            // 水平Divider
            Text("Section 1")
            Divider()
            Text("Section 2")

            Spacer()

            // カスタムDivider
            VStack(spacing: 20) {
                Text("Item 1")
                CustomDivider()
                Text("Item 2")
                CustomDivider()
                Text("Item 3")
            }
        }
        .padding()
    }
}

struct CustomDivider: View {
    var body: some View {
        Rectangle()
            .fill(.gray.opacity(0.3))
            .frame(height: 1)
            .padding(.horizontal)
    }
}
```

## フレームとサイズ調整

### 固定サイズ vs 可変サイズ

```swift
struct FrameExampleView: View {
    var body: some View {
        VStack(spacing: 20) {
            // 固定サイズ
            Text("Fixed")
                .frame(width: 200, height: 100)
                .background(.blue)

            // 最小サイズ
            Text("Minimum")
                .frame(minWidth: 100, minHeight: 50)
                .background(.green)

            // 最大サイズ
            Text("Maximum")
                .frame(maxWidth: .infinity, maxHeight: 100)
                .background(.orange)

            // 理想サイズ
            Text("Ideal")
                .frame(idealWidth: 200, idealHeight: 100)
                .background(.purple)

            // 全ての組み合わせ
            Text("Combined")
                .frame(
                    minWidth: 100,
                    idealWidth: 200,
                    maxWidth: .infinity,
                    minHeight: 50,
                    maxHeight: 100
                )
                .background(.red)
        }
        .padding()
    }
}
```

### アスペクト比の維持

```swift
struct AspectRatioView: View {
    var body: some View {
        VStack(spacing: 20) {
            // 16:9 アスペクト比
            Color.blue
                .aspectRatio(16/9, contentMode: .fit)
                .frame(maxWidth: .infinity)

            // 1:1 (正方形)
            AsyncImage(url: URL(string: "https://via.placeholder.com/300")) { image in
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } placeholder: {
                Color.gray
            }
            .aspectRatio(1, contentMode: .fit)
            .frame(width: 200)
            .clipShape(RoundedRectangle(cornerRadius: 12))

            // 4:3 アスペクト比でfill
            Color.green
                .aspectRatio(4/3, contentMode: .fill)
                .frame(height: 200)
                .clipped()
        }
        .padding()
    }
}
```

## Alignment と Padding

### Alignmentの詳細

```swift
struct AlignmentExampleView: View {
    var body: some View {
        VStack(spacing: 30) {
            // HStackのalignment
            Group {
                HStack(alignment: .top) {
                    rectangle("Top")
                    tallRectangle()
                    rectangle("Top")
                }

                HStack(alignment: .center) {
                    rectangle("Center")
                    tallRectangle()
                    rectangle("Center")
                }

                HStack(alignment: .bottom) {
                    rectangle("Bottom")
                    tallRectangle()
                    rectangle("Bottom")
                }
            }

            Divider()

            // VStackのalignment
            Group {
                VStack(alignment: .leading) {
                    Text("Leading alignment")
                    Text("Short")
                    Text("Very long text here")
                }
                .frame(maxWidth: .infinity)
                .background(.gray.opacity(0.2))

                VStack(alignment: .trailing) {
                    Text("Trailing alignment")
                    Text("Short")
                    Text("Very long text here")
                }
                .frame(maxWidth: .infinity)
                .background(.gray.opacity(0.2))
            }
        }
        .padding()
    }

    func rectangle(_ text: String) -> some View {
        Text(text)
            .frame(width: 80, height: 50)
            .background(.blue)
            .foregroundColor(.white)
    }

    func tallRectangle() -> some View {
        Rectangle()
            .fill(.red)
            .frame(width: 80, height: 100)
    }
}
```

### カスタムAlignment

```swift
extension VerticalAlignment {
    private struct CustomAlignment: AlignmentID {
        static func defaultValue(in context: ViewDimensions) -> CGFloat {
            context[.bottom]
        }
    }

    static let custom = VerticalAlignment(CustomAlignment.self)
}

struct CustomAlignmentView: View {
    var body: some View {
        HStack(alignment: .custom) {
            VStack {
                Text("Left")
                Text("Content")
                    .alignmentGuide(.custom) { d in d[.bottom] }
            }

            VStack {
                Text("Right")
                Text("Aligned")
                    .alignmentGuide(.custom) { d in d[.bottom] }
                Text("Content")
            }
        }
        .padding()
    }
}
```

### Paddingのパターン

```swift
struct PaddingExampleView: View {
    var body: some View {
        VStack(spacing: 20) {
            // 全方向に同じpadding
            Text("All sides: 20")
                .padding(20)
                .background(.blue)

            // 個別指定
            Text("Custom")
                .padding(.leading, 40)
                .padding(.trailing, 10)
                .padding(.vertical, 20)
                .background(.green)

            // EdgeInsetsで指定
            Text("EdgeInsets")
                .padding(EdgeInsets(top: 10, leading: 20, bottom: 10, trailing: 20))
                .background(.orange)

            // 複数のpadding適用
            Text("Layered")
                .padding()
                .background(.red)
                .padding()
                .background(.blue)
                .padding()
                .background(.green)
        }
        .padding()
    }
}
```

## GeometryReader

### 基本的な使用

```swift
struct GeometryReaderBasicView: View {
    var body: some View {
        GeometryReader { geometry in
            VStack {
                Text("Width: \(Int(geometry.size.width))")
                Text("Height: \(Int(geometry.size.height))")

                Rectangle()
                    .fill(.blue)
                    .frame(
                        width: geometry.size.width * 0.8,
                        height: geometry.size.height * 0.3
                    )
            }
            .frame(
                width: geometry.size.width,
                height: geometry.size.height
            )
        }
    }
}
```

### 実践的な使用例

```swift
struct AdaptiveCardView: View {
    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width

            if width > 600 {
                // iPad / 横向き
                HStack(spacing: 20) {
                    ImageSection()
                        .frame(width: width * 0.4)
                    ContentSection()
                        .frame(width: width * 0.6)
                }
            } else {
                // iPhone / 縦向き
                VStack(spacing: 20) {
                    ImageSection()
                        .frame(height: 200)
                    ContentSection()
                }
            }
        }
        .padding()
    }
}

struct ImageSection: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 12)
            .fill(.blue)
            .overlay {
                Image(systemName: "photo")
                    .font(.largeTitle)
                    .foregroundColor(.white)
            }
    }
}

struct ContentSection: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Title")
                .font(.title)
                .fontWeight(.bold)
            Text("Description goes here. This is some sample text to demonstrate the layout.")
                .font(.body)
                .foregroundColor(.secondary)
        }
    }
}
```

### GeometryReaderの注意点

```swift
struct GeometryReaderPitfallsView: View {
    var body: some View {
        VStack {
            // ❌ 不必要なGeometryReader
            // GeometryReader { geometry in
            //     Text("Hello")
            //         .frame(width: geometry.size.width)
            // }

            // ✅ より良い方法
            Text("Hello")
                .frame(maxWidth: .infinity)

            // ❌ GeometryReaderは利用可能な全スペースを占有する
            // GeometryReader { geometry in
            //     Text("Small Text")
            // }

            // ✅ backgroundやoverlayで使用
            Text("Small Text")
                .background(
                    GeometryReader { geometry in
                        Color.clear
                            .onAppear {
                                print("Size: \(geometry.size)")
                            }
                    }
                )
        }
    }
}
```

## Custom Layout (iOS 16+)

### 基本的なカスタムレイアウト

```swift
struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = computeRows(proposal: proposal, subviews: subviews)
        let width = proposal.width ?? 0
        let height = rows.reduce(0) { $0 + $1.maxHeight } + CGFloat(rows.count - 1) * spacing
        return CGSize(width: width, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let rows = computeRows(proposal: proposal, subviews: subviews)
        var y = bounds.minY

        for row in rows {
            var x = bounds.minX

            for index in row.indices {
                let subview = subviews[index]
                let size = subview.sizeThatFits(.unspecified)

                subview.place(
                    at: CGPoint(x: x, y: y),
                    proposal: ProposedViewSize(size)
                )

                x += size.width + spacing
            }

            y += row.maxHeight + spacing
        }
    }

    private func computeRows(proposal: ProposedViewSize, subviews: Subviews) -> [Row] {
        var rows: [Row] = []
        var currentRow = Row()
        var x: CGFloat = 0
        let width = proposal.width ?? .infinity

        for (index, subview) in subviews.enumerated() {
            let size = subview.sizeThatFits(.unspecified)

            if x + size.width > width && !currentRow.indices.isEmpty {
                rows.append(currentRow)
                currentRow = Row()
                x = 0
            }

            currentRow.indices.append(index)
            currentRow.maxHeight = max(currentRow.maxHeight, size.height)
            x += size.width + spacing
        }

        if !currentRow.indices.isEmpty {
            rows.append(currentRow)
        }

        return rows
    }

    struct Row {
        var indices: [Int] = []
        var maxHeight: CGFloat = 0
    }
}

struct FlowLayoutExampleView: View {
    let tags = ["SwiftUI", "iOS", "Xcode", "Swift", "Design", "Development", "Mobile", "App"]

    var body: some View {
        FlowLayout(spacing: 8) {
            ForEach(tags, id: \.self) { tag in
                Text(tag)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.blue.opacity(0.2))
                    .cornerRadius(16)
            }
        }
        .padding()
    }
}
```

## LazyStacks と ScrollView

### LazyVStack

```swift
struct LazyVStackExampleView: View {
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                ForEach(0..<100) { index in
                    ItemRow(index: index)
                        .onAppear {
                            print("Item \(index) appeared")
                        }
                }
            }
            .padding()
        }
    }
}

struct ItemRow: View {
    let index: Int

    var body: some View {
        HStack {
            Circle()
                .fill(.blue)
                .frame(width: 50, height: 50)

            VStack(alignment: .leading) {
                Text("Item \(index)")
                    .font(.headline)
                Text("Description for item \(index)")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Image(systemName: "chevron.right")
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}
```

### LazyHStack

```swift
struct LazyHStackExampleView: View {
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            LazyHStack(spacing: 16) {
                ForEach(0..<20) { index in
                    CardView(index: index)
                }
            }
            .padding()
        }
    }
}

struct CardView: View {
    let index: Int

    var body: some View {
        VStack {
            RoundedRectangle(cornerRadius: 12)
                .fill(.blue.gradient)
                .frame(width: 200, height: 250)
                .overlay {
                    Text("Card \(index)")
                        .font(.title)
                        .foregroundColor(.white)
                }
        }
    }
}
```

### ページング

```swift
struct PagingScrollView: View {
    @State private var currentPage = 0
    let colors: [Color] = [.red, .blue, .green, .orange, .purple]

    var body: some View {
        VStack {
            ScrollView(.horizontal) {
                LazyHStack(spacing: 0) {
                    ForEach(0..<colors.count, id: \.self) { index in
                        colors[index]
                            .containerRelativeFrame(.horizontal)
                            .id(index)
                    }
                }
                .scrollTargetLayout()
            }
            .scrollTargetBehavior(.paging)
            .frame(height: 300)

            // ページインジケーター
            HStack(spacing: 8) {
                ForEach(0..<colors.count, id: \.self) { index in
                    Circle()
                        .fill(index == currentPage ? .primary : .secondary)
                        .frame(width: 8, height: 8)
                }
            }
            .padding()
        }
    }
}
```

### Pull to Refresh

```swift
struct RefreshableListView: View {
    @State private var items: [String] = Array(0..<20).map { "Item \($0)" }
    @State private var isRefreshing = false

    var body: some View {
        List(items, id: \.self) { item in
            Text(item)
        }
        .refreshable {
            await refresh()
        }
    }

    func refresh() async {
        isRefreshing = true
        try? await Task.sleep(nanoseconds: 2_000_000_000)

        // 新しいアイテムを追加
        items = Array(0..<20).map { "Item \($0) - Updated" }
        isRefreshing = false
    }
}
```

## Grid レイアウト

### LazyVGrid

```swift
struct GridExampleView: View {
    let columns = [
        GridItem(.flexible()),
        GridItem(.flexible()),
        GridItem(.flexible())
    ]

    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 16) {
                ForEach(0..<20) { index in
                    GridItemView(index: index)
                }
            }
            .padding()
        }
    }
}

struct GridItemView: View {
    let index: Int

    var body: some View {
        VStack {
            RoundedRectangle(cornerRadius: 12)
                .fill(.blue.gradient)
                .aspectRatio(1, contentMode: .fit)
                .overlay {
                    Text("\(index)")
                        .font(.title)
                        .foregroundColor(.white)
                }
        }
    }
}
```

### アダプティブGrid

```swift
struct AdaptiveGridView: View {
    let columns = [
        GridItem(.adaptive(minimum: 100))
    ]

    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 16) {
                ForEach(0..<30) { index in
                    RoundedRectangle(cornerRadius: 12)
                        .fill(.blue)
                        .frame(height: 100)
                        .overlay {
                            Text("\(index)")
                                .foregroundColor(.white)
                        }
                }
            }
            .padding()
        }
    }
}
```

### Grid (iOS 16+)

```swift
struct ModernGridView: View {
    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 16) {
            GridRow {
                Text("Name")
                    .fontWeight(.bold)
                Text("Age")
                    .fontWeight(.bold)
                Text("City")
                    .fontWeight(.bold)
            }

            Divider()

            GridRow {
                Text("John")
                Text("28")
                Text("New York")
            }

            GridRow {
                Text("Alice")
                Text("32")
                Text("San Francisco")
            }

            GridRow {
                Text("Bob")
                Text("25")
                Text("Seattle")
            }
        }
        .padding()
    }
}
```

## NavigationStack (iOS 16+)

### 基本的なナビゲーション

```swift
struct NavigationExampleView: View {
    var body: some View {
        NavigationStack {
            List(0..<20) { index in
                NavigationLink("Item \(index)", value: index)
            }
            .navigationDestination(for: Int.self) { value in
                DetailView(number: value)
            }
            .navigationTitle("List")
        }
    }
}

struct DetailView: View {
    let number: Int

    var body: some View {
        VStack {
            Text("Detail for item \(number)")
                .font(.title)
            NavigationLink("Go Deeper", value: number + 1)
        }
        .navigationTitle("Detail \(number)")
        .navigationBarTitleDisplayMode(.inline)
    }
}
```

### プログラマティックナビゲーション

```swift
enum Route: Hashable {
    case home
    case profile(userId: String)
    case settings
    case detail(id: Int)
}

struct ProgrammaticNavigationView: View {
    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
            VStack(spacing: 20) {
                Button("Go to Profile") {
                    path.append(Route.profile(userId: "user123"))
                }

                Button("Go to Settings") {
                    path.append(Route.settings)
                }

                Button("Deep Link") {
                    path.append(Route.profile(userId: "user456"))
                    path.append(Route.detail(id: 1))
                }

                Button("Pop to Root") {
                    path.removeLast(path.count)
                }
            }
            .navigationTitle("Home")
            .navigationDestination(for: Route.self) { route in
                destinationView(for: route)
            }
        }
    }

    @ViewBuilder
    func destinationView(for route: Route) -> some View {
        switch route {
        case .home:
            Text("Home")
        case .profile(let userId):
            ProfileView(userId: userId, path: $path)
        case .settings:
            SettingsView()
        case .detail(let id):
            Text("Detail \(id)")
        }
    }
}

struct ProfileView: View {
    let userId: String
    @Binding var path: NavigationPath

    var body: some View {
        VStack {
            Text("Profile: \(userId)")
            Button("Go to Settings") {
                path.append(Route.settings)
            }
        }
        .navigationTitle("Profile")
    }
}

struct SettingsView: View {
    var body: some View {
        Text("Settings")
            .navigationTitle("Settings")
    }
}
```

### NavigationSplitView対応

```swift
struct SplitNavigationView: View {
    @State private var selectedCategory: String?
    @State private var selectedItem: String?

    let categories = ["Electronics", "Books", "Clothing"]
    let items = ["Item 1", "Item 2", "Item 3"]

    var body: some View {
        NavigationSplitView {
            // Sidebar
            List(categories, id: \.self, selection: $selectedCategory) { category in
                Text(category)
            }
            .navigationTitle("Categories")
        } content: {
            // Content
            if let category = selectedCategory {
                List(items, id: \.self, selection: $selectedItem) { item in
                    Text(item)
                }
                .navigationTitle(category)
            } else {
                Text("Select a category")
            }
        } detail: {
            // Detail
            if let item = selectedItem {
                Text("Detail for \(item)")
            } else {
                Text("Select an item")
            }
        }
    }
}
```

## Modal Presentation

### Sheet

```swift
struct SheetExampleView: View {
    @State private var isShowingSheet = false
    @State private var selectedItem: String?

    var body: some View {
        VStack(spacing: 20) {
            Button("Show Sheet") {
                isShowingSheet = true
            }

            Button("Show Item Sheet") {
                selectedItem = "Item 1"
            }
        }
        .sheet(isPresented: $isShowingSheet) {
            SheetContentView()
        }
        .sheet(item: $selectedItem) { item in
            ItemDetailSheet(item: item)
        }
    }
}

struct SheetContentView: View {
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationStack {
            VStack {
                Text("Sheet Content")
                Button("Close") {
                    dismiss()
                }
            }
            .navigationTitle("Sheet")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }
}

struct ItemDetailSheet: View {
    let item: String
    @Environment(\.dismiss) var dismiss

    var body: some View {
        VStack {
            Text("Detail for \(item)")
            Button("Close") {
                dismiss()
            }
        }
    }
}

extension String: Identifiable {
    public var id: String { self }
}
```

### FullScreenCover

```swift
struct FullScreenCoverExampleView: View {
    @State private var isShowingFullScreen = false

    var body: some View {
        Button("Show Full Screen") {
            isShowingFullScreen = true
        }
        .fullScreenCover(isPresented: $isShowingFullScreen) {
            FullScreenContentView()
        }
    }
}

struct FullScreenContentView: View {
    @Environment(\.dismiss) var dismiss

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack {
                Text("Full Screen Content")
                    .font(.largeTitle)
                    .foregroundColor(.white)

                Button("Close") {
                    dismiss()
                }
                .foregroundColor(.white)
                .padding()
            }
        }
    }
}
```

### Popover

```swift
struct PopoverExampleView: View {
    @State private var isShowingPopover = false

    var body: some View {
        Button("Show Popover") {
            isShowingPopover = true
        }
        .popover(isPresented: $isShowingPopover) {
            PopoverContentView()
                .presentationCompactAdaptation(.popover)
        }
    }
}

struct PopoverContentView: View {
    var body: some View {
        VStack(spacing: 20) {
            Text("Popover Content")
                .font(.headline)
            Text("This is displayed in a popover")
            Button("Action") {
                // アクション
            }
        }
        .padding()
        .frame(width: 300, height: 200)
    }
}
```

### Alert と ConfirmationDialog

```swift
struct AlertExampleView: View {
    @State private var showingAlert = false
    @State private var showingConfirmation = false

    var body: some View {
        VStack(spacing: 20) {
            Button("Show Alert") {
                showingAlert = true
            }
            .alert("Title", isPresented: $showingAlert) {
                Button("OK", role: .cancel) { }
                Button("Delete", role: .destructive) {
                    // 削除処理
                }
            } message: {
                Text("This is an alert message")
            }

            Button("Show Confirmation") {
                showingConfirmation = true
            }
            .confirmationDialog("Select an option", isPresented: $showingConfirmation) {
                Button("Option 1") { }
                Button("Option 2") { }
                Button("Cancel", role: .cancel) { }
                Button("Delete", role: .destructive) { }
            }
        }
    }
}
```

## TabView

### 基本的なTabView

```swift
struct TabViewExampleView: View {
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            HomeTab()
                .tabItem {
                    Label("Home", systemImage: "house")
                }
                .tag(0)

            SearchTab()
                .tabItem {
                    Label("Search", systemImage: "magnifyingglass")
                }
                .tag(1)

            ProfileTab()
                .tabItem {
                    Label("Profile", systemImage: "person")
                }
                .tag(2)
        }
    }
}

struct HomeTab: View {
    var body: some View {
        NavigationStack {
            Text("Home")
                .navigationTitle("Home")
        }
    }
}

struct SearchTab: View {
    var body: some View {
        NavigationStack {
            Text("Search")
                .navigationTitle("Search")
        }
    }
}

struct ProfileTab: View {
    var body: some View {
        NavigationStack {
            Text("Profile")
                .navigationTitle("Profile")
        }
    }
}
```

### バッジ付きTab

```swift
struct BadgedTabView: View {
    @State private var notificationCount = 5

    var body: some View {
        TabView {
            Text("Home")
                .tabItem {
                    Label("Home", systemImage: "house")
                }

            Text("Messages")
                .tabItem {
                    Label("Messages", systemImage: "message")
                }
                .badge(notificationCount)

            Text("Settings")
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .badge("New")
        }
    }
}
```

## SplitView

### 2カラムSplitView

```swift
struct TwoColumnSplitView: View {
    @State private var selectedItem: String?

    let items = Array(0..<20).map { "Item \($0)" }

    var body: some View {
        NavigationSplitView {
            List(items, id: \.self, selection: $selectedItem) { item in
                Text(item)
            }
            .navigationTitle("List")
        } detail: {
            if let item = selectedItem {
                ItemDetailView(item: item)
            } else {
                Text("Select an item")
                    .foregroundColor(.secondary)
            }
        }
    }
}

struct ItemDetailView: View {
    let item: String

    var body: some View {
        VStack {
            Text("Detail for \(item)")
                .font(.title)
            Spacer()
        }
        .navigationTitle(item)
    }
}
```

### 3カラムSplitView

```swift
struct ThreeColumnSplitView: View {
    @State private var selectedCategory: String?
    @State private var selectedItem: String?

    let categories = ["Category 1", "Category 2", "Category 3"]

    var body: some View {
        NavigationSplitView {
            // Sidebar
            List(categories, id: \.self, selection: $selectedCategory) { category in
                Text(category)
            }
            .navigationTitle("Categories")
        } content: {
            // Content
            if let category = selectedCategory {
                List(0..<10, id: \.self, selection: $selectedItem) { index in
                    Text("Item \(index)")
                }
                .navigationTitle(category)
            }
        } detail: {
            // Detail
            if let item = selectedItem {
                Text("Detail for \(item)")
                    .font(.title)
            } else {
                Text("Select an item")
                    .foregroundColor(.secondary)
            }
        }
    }
}
```

## アダプティブレイアウト

### 環境値による分岐

```swift
struct AdaptiveLayoutView: View {
    @Environment(\.horizontalSizeClass) var horizontalSizeClass
    @Environment(\.verticalSizeClass) var verticalSizeClass

    var body: some View {
        Group {
            if horizontalSizeClass == .compact {
                CompactLayout()
            } else {
                RegularLayout()
            }
        }
    }
}

struct CompactLayout: View {
    var body: some View {
        VStack {
            HeaderView()
            ContentView()
        }
    }
}

struct RegularLayout: View {
    var body: some View {
        HStack {
            Sidebar()
            ContentView()
        }
    }
}

struct HeaderView: View {
    var body: some View {
        Text("Header")
            .font(.title)
            .padding()
    }
}

struct ContentView: View {
    var body: some View {
        Text("Content")
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(.systemGray6))
    }
}

struct Sidebar: View {
    var body: some View {
        VStack {
            Text("Sidebar")
            Spacer()
        }
        .frame(width: 200)
        .background(Color(.systemGray5))
    }
}
```

### ViewThatFits (iOS 16+)

```swift
struct ViewThatFitsExampleView: View {
    var body: some View {
        ViewThatFits {
            // 最初に試すレイアウト
            HStack {
                ForEach(0..<5) { index in
                    CardView(title: "Card \(index)")
                }
            }

            // 収まらない場合のレイアウト
            VStack {
                ForEach(0..<5) { index in
                    CardView(title: "Card \(index)")
                }
            }
        }
        .padding()
    }
}

struct CardView: View {
    let title: String

    var body: some View {
        Text(title)
            .padding()
            .background(.blue)
            .foregroundColor(.white)
            .cornerRadius(8)
    }
}
```

## よくある問題と解決策

### 問題1: GeometryReaderが予期しないスペースを取る

```swift
// ❌ 問題のあるコード
struct BadGeometryView: View {
    var body: some View {
        VStack {
            Text("Top")
            GeometryReader { geometry in
                Text("Middle")
            }
            Text("Bottom")
        }
    }
}

// ✅ 改善したコード
struct GoodGeometryView: View {
    var body: some View {
        VStack {
            Text("Top")
            Text("Middle")
                .background(
                    GeometryReader { geometry in
                        Color.clear
                            .preference(key: SizePreferenceKey.self, value: geometry.size)
                    }
                )
            Text("Bottom")
        }
    }
}

struct SizePreferenceKey: PreferenceKey {
    static var defaultValue: CGSize = .zero
    static func reduce(value: inout CGSize, nextValue: () -> CGSize) {
        value = nextValue()
    }
}
```

### 問題2: LazyStackを使うべきか判断できない

```swift
// ✅ 判断基準:
// - アイテム数が多い (>50) → LazyStack
// - アイテム数が少ない (<20) → 通常のStack
// - スクロールが必要 → LazyStack + ScrollView

// 多数のアイテム
ScrollView {
    LazyVStack {
        ForEach(0..<1000) { index in
            Text("Item \(index)")
        }
    }
}

// 少数のアイテム
VStack {
    ForEach(0..<10) { index in
        Text("Item \(index)")
    }
}
```

### 問題3: NavigationStackでの状態管理

```swift
// ✅ pathを上位で管理してディープリンクに対応
@main
struct App: App {
    @State private var navigationPath = NavigationPath()

    var body: some Scene {
        WindowGroup {
            NavigationStack(path: $navigationPath) {
                RootView()
                    .onOpenURL { url in
                        handleDeepLink(url)
                    }
            }
        }
    }

    func handleDeepLink(_ url: URL) {
        // URLからpathを構築
    }
}
```

---

**関連ガイド:**
- [01-state-management.md](./01-state-management.md) - 状態管理パターン
- [03-performance-best-practices.md](./03-performance-best-practices.md) - パフォーマンス最適化

**関連Skills:**
- [ios-development](../../ios-development/SKILL.md) - iOS開発全般
- [frontend-performance](../../frontend-performance/SKILL.md) - パフォーマンス最適化

**参考資料:**
- [SwiftUI Layout System](https://developer.apple.com/documentation/swiftui/view-layout)
- [WWDC - Compose custom layouts with SwiftUI](https://developer.apple.com/videos/play/wwdc2022/10056/)

**更新履歴:**
- 2025-12-30: 初版作成

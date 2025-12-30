# Performance & Best Practices - SwiftUIパフォーマンス最適化とベストプラクティス

## 📋 目次

1. [概要](#概要)
2. [パフォーマンス最適化の基本原則](#パフォーマンス最適化の基本原則)
3. [不要な再描画の防止](#不要な再描画の防止)
4. [@Published の最適化](#published-の最適化)
5. [LazyLoading の活用](#lazyloading-の活用)
6. [画像の最適化](#画像の最適化)
7. [アニメーションの最適化](#アニメーションの最適化)
8. [メモリ管理](#メモリ管理)
9. [非同期処理のベストプラクティス](#非同期処理のベストプラクティス)
10. [プレビューの最適化](#プレビューの最適化)
11. [デバッグとプロファイリング](#デバッグとプロファイリング)
12. [アクセシビリティベストプラクティス](#アクセシビリティベストプラクティス)
13. [コードの可読性と保守性](#コードの可読性と保守性)
14. [テストのベストプラクティス](#テストのベストプラクティス)
15. [よくある問題と解決策](#よくある問題と解決策)

## 概要

SwiftUIアプリのパフォーマンスを最大化し、保守性の高いコードを書くためのベストプラクティスを解説します。

### このガイドの対象者

- SwiftUI中級〜上級者
- パフォーマンス改善を行いたい開発者
- 大規模プロジェクトを構築する開発者

### 学べること

- 効率的なView更新戦略
- メモリ使用量の最適化
- スムーズなアニメーション実装
- 保守性の高いコード設計

## パフォーマンス最適化の基本原則

### 測定してから最適化する

```swift
// ✅ Instruments を使ってボトルネックを特定
// - Time Profiler: CPU使用率
// - Allocations: メモリ使用量
// - SwiftUI: View更新頻度

struct PerformanceMonitoringView: View {
    @State private var items: [Item] = []

    var body: some View {
        List(items) { item in
            ItemRow(item: item)
        }
        .task {
            let start = Date()
            await loadItems()
            let duration = Date().timeIntervalSince(start)
            print("Load time: \(duration)s")
        }
    }

    func loadItems() async {
        // データ読み込み
    }
}

struct Item: Identifiable {
    let id: UUID
    let title: String
}

struct ItemRow: View {
    let item: Item
    var body: some View { Text(item.title) }
}
```

### View階層の最適化

```swift
// ❌ 深すぎるView階層
struct BadHierarchyView: View {
    var body: some View {
        VStack {
            VStack {
                VStack {
                    VStack {
                        Text("Content")
                    }
                }
            }
        }
    }
}

// ✅ フラットな階層
struct GoodHierarchyView: View {
    var body: some View {
        VStack {
            Text("Content")
        }
        .padding()
    }
}
```

### 計算の重複を避ける

```swift
// ❌ 重複計算
struct BadCalculationView: View {
    let items: [Int]

    var body: some View {
        VStack {
            Text("Sum: \(items.reduce(0, +))")
            Text("Average: \(Double(items.reduce(0, +)) / Double(items.count))")
            Text("Max: \(items.max() ?? 0)")
        }
    }
}

// ✅ 計算を一度だけ実行
struct GoodCalculationView: View {
    let items: [Int]

    private var sum: Int {
        items.reduce(0, +)
    }

    private var average: Double {
        Double(sum) / Double(items.count)
    }

    private var maximum: Int {
        items.max() ?? 0
    }

    var body: some View {
        VStack {
            Text("Sum: \(sum)")
            Text("Average: \(average)")
            Text("Max: \(maximum)")
        }
    }
}
```

## 不要な再描画の防止

### Equatable の活用

```swift
// ❌ 親が更新されるたびに子も更新される
struct ParentView: View {
    @State private var counter = 0
    let data = ExpensiveData()

    var body: some View {
        VStack {
            Button("Counter: \(counter)") {
                counter += 1
            }
            ExpensiveChildView(data: data) // 毎回再描画される
        }
    }
}

struct ExpensiveChildView: View {
    let data: ExpensiveData

    var body: some View {
        // 重い描画処理
        Text("Expensive render")
            .onAppear {
                print("Child rendered") // counter変更のたびに呼ばれる
            }
    }
}

// ✅ Equatableで再描画を制御
struct OptimizedParentView: View {
    @State private var counter = 0
    let data = ExpensiveData()

    var body: some View {
        VStack {
            Button("Counter: \(counter)") {
                counter += 1
            }
            OptimizedChildView(data: data)
                .equatable() // dataが変わらない限り再描画しない
        }
    }
}

struct OptimizedChildView: View, Equatable {
    let data: ExpensiveData

    var body: some View {
        Text("Expensive render")
            .onAppear {
                print("Child rendered") // 初回のみ
            }
    }

    static func == (lhs: OptimizedChildView, rhs: OptimizedChildView) -> Bool {
        lhs.data.id == rhs.data.id
    }
}

struct ExpensiveData: Identifiable {
    let id = UUID()
    let values: [Double] = Array(repeating: 0, count: 1000)
}
```

### @State の適切な配置

```swift
// ❌ 不要な@State
struct BadStateView: View {
    @State private var staticText = "Hello" // 変更されない値に@Stateは不要

    var body: some View {
        Text(staticText)
    }
}

// ✅ 必要な場合のみ@State
struct GoodStateView: View {
    let staticText = "Hello" // 定数で十分
    @State private var counter = 0 // 変更される値のみ@State

    var body: some View {
        VStack {
            Text(staticText)
            Text("Counter: \(counter)")
            Button("Increment") {
                counter += 1
            }
        }
    }
}
```

### ViewBuilderの最適化

```swift
// ❌ 非効率なViewBuilder
struct BadViewBuilderView: View {
    let items: [String]

    var body: some View {
        VStack {
            // items全体が変わらなくても全て再構築される
            ForEach(items.indices, id: \.self) { index in
                Text(items[index])
            }
        }
    }
}

// ✅ 効率的なViewBuilder
struct GoodViewBuilderView: View {
    let items: [Item]

    var body: some View {
        VStack {
            // Identifiableなアイテムを使用
            ForEach(items) { item in
                ItemView(item: item)
                    .id(item.id) // 明示的なID
            }
        }
    }
}

struct ItemView: View, Equatable {
    let item: Item

    var body: some View {
        Text(item.title)
    }

    static func == (lhs: ItemView, rhs: ItemView) -> Bool {
        lhs.item.id == rhs.item.id
    }
}
```

## @Published の最適化

### 必要最小限の@Published

```swift
// ❌ 過剰な@Published
class BadViewModel: ObservableObject {
    @Published var tempValue1 = 0
    @Published var tempValue2 = 0
    @Published var tempValue3 = 0
    @Published var displayText = ""

    func updateDisplay() {
        tempValue1 += 1
        tempValue2 += 2
        tempValue3 += 3
        displayText = "\(tempValue1 + tempValue2 + tempValue3)"
    }
}

// ✅ UI表示用のプロパティのみ@Published
class GoodViewModel: ObservableObject {
    private var tempValue1 = 0
    private var tempValue2 = 0
    private var tempValue3 = 0

    @Published var displayText = ""

    func updateDisplay() {
        tempValue1 += 1
        tempValue2 += 2
        tempValue3 += 3
        displayText = "\(tempValue1 + tempValue2 + tempValue3)"
    }
}
```

### バッチ更新

```swift
class OptimizedViewModel: ObservableObject {
    @Published var items: [Item] = []
    private var updateTimer: Timer?

    func startUpdates() {
        var pendingItems: [Item] = []

        updateTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            // 0.1秒ごとに大量のアイテムを受信
            let newItems = self.generateItems(count: 100)
            pendingItems.append(contentsOf: newItems)

            // 1秒に1回だけUIを更新
            if pendingItems.count >= 1000 {
                self.items.append(contentsOf: pendingItems)
                pendingItems.removeAll()
            }
        }
    }

    private func generateItems(count: Int) -> [Item] {
        (0..<count).map { Item(id: UUID(), title: "Item \($0)") }
    }
}
```

### objectWillChange の手動制御

```swift
class ManualUpdateViewModel: ObservableObject {
    private var internalCounter = 0

    var displayValue: String {
        "Count: \(internalCounter)"
    }

    func increment() {
        internalCounter += 1

        // 10回に1回だけUI更新を通知
        if internalCounter % 10 == 0 {
            objectWillChange.send()
        }
    }
}

struct ManualUpdateView: View {
    @StateObject private var viewModel = ManualUpdateViewModel()

    var body: some View {
        VStack {
            Text(viewModel.displayValue)
            Button("Increment") {
                viewModel.increment()
            }
        }
    }
}
```

## LazyLoading の活用

### LazyVStack vs VStack

```swift
// ❌ 全てのアイテムを一度に作成
struct EagerLoadingView: View {
    let items = Array(0..<1000)

    var body: some View {
        ScrollView {
            VStack {
                ForEach(items, id: \.self) { item in
                    ExpensiveRow(item: item)
                }
            }
        }
    }
}

// ✅ 表示されるアイテムのみ作成
struct LazyLoadingView: View {
    let items = Array(0..<1000)

    var body: some View {
        ScrollView {
            LazyVStack {
                ForEach(items, id: \.self) { item in
                    ExpensiveRow(item: item)
                }
            }
        }
    }
}

struct ExpensiveRow: View {
    let item: Int

    var body: some View {
        HStack {
            Circle()
                .fill(.blue)
                .frame(width: 50, height: 50)
            Text("Item \(item)")
            Spacer()
        }
        .padding()
        .onAppear {
            print("Row \(item) appeared")
        }
    }
}
```

### ページネーション

```swift
class PaginatedViewModel: ObservableObject {
    @Published var items: [Item] = []
    @Published var isLoading = false

    private var currentPage = 0
    private let pageSize = 20

    func loadMore() async {
        guard !isLoading else { return }

        await MainActor.run {
            isLoading = true
        }

        // API呼び出し
        try? await Task.sleep(nanoseconds: 500_000_000)
        let newItems = generateItems(page: currentPage, size: pageSize)

        await MainActor.run {
            self.items.append(contentsOf: newItems)
            self.currentPage += 1
            self.isLoading = false
        }
    }

    private func generateItems(page: Int, size: Int) -> [Item] {
        let start = page * size
        return (start..<start+size).map {
            Item(id: UUID(), title: "Item \($0)")
        }
    }
}

struct PaginatedListView: View {
    @StateObject private var viewModel = PaginatedViewModel()

    var body: some View {
        ScrollView {
            LazyVStack {
                ForEach(viewModel.items) { item in
                    Text(item.title)
                        .onAppear {
                            if item == viewModel.items.last {
                                Task {
                                    await viewModel.loadMore()
                                }
                            }
                        }
                }

                if viewModel.isLoading {
                    ProgressView()
                }
            }
        }
        .task {
            await viewModel.loadMore()
        }
    }
}
```

### プリロード戦略

```swift
struct PreloadingListView: View {
    @StateObject private var viewModel = PreloadViewModel()
    private let preloadThreshold = 5 // 最後の5件前でプリロード開始

    var body: some View {
        List(viewModel.items) { item in
            ItemRow(item: item)
                .onAppear {
                    if shouldPreload(item: item) {
                        Task {
                            await viewModel.loadMore()
                        }
                    }
                }
        }
    }

    private func shouldPreload(item: Item) -> Bool {
        guard let index = viewModel.items.firstIndex(where: { $0.id == item.id }) else {
            return false
        }
        return index >= viewModel.items.count - preloadThreshold
    }
}

class PreloadViewModel: ObservableObject {
    @Published var items: [Item] = []

    func loadMore() async {
        // ロード処理
    }
}
```

## 画像の最適化

### AsyncImage の効率的な使用

```swift
struct OptimizedImageView: View {
    let url: URL

    var body: some View {
        AsyncImage(url: url) { phase in
            switch phase {
            case .empty:
                ProgressView()
            case .success(let image):
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            case .failure:
                Image(systemName: "photo")
                    .foregroundColor(.gray)
            @unknown default:
                EmptyView()
            }
        }
        .frame(width: 200, height: 200)
        .clipped()
    }
}
```

### 画像キャッシュの実装

```swift
actor ImageCache {
    static let shared = ImageCache()

    private var cache: [URL: Image] = [:]

    func image(for url: URL) -> Image? {
        cache[url]
    }

    func set(_ image: Image, for url: URL) {
        cache[url] = image
    }

    func clear() {
        cache.removeAll()
    }
}

class ImageLoader: ObservableObject {
    @Published var image: Image?
    @Published var isLoading = false

    func load(from url: URL) async {
        isLoading = true
        defer { isLoading = false }

        // キャッシュチェック
        if let cached = await ImageCache.shared.image(for: url) {
            self.image = cached
            return
        }

        // ダウンロード
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            if let uiImage = UIImage(data: data) {
                let image = Image(uiImage: uiImage)
                await ImageCache.shared.set(image, for: url)
                await MainActor.run {
                    self.image = image
                }
            }
        } catch {
            print("Failed to load image: \(error)")
        }
    }
}

struct CachedImageView: View {
    let url: URL
    @StateObject private var loader = ImageLoader()

    var body: some View {
        Group {
            if let image = loader.image {
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else if loader.isLoading {
                ProgressView()
            } else {
                Color.gray
            }
        }
        .task {
            await loader.load(from: url)
        }
    }
}
```

### サムネイル生成

```swift
extension UIImage {
    func thumbnail(size: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { context in
            self.draw(in: CGRect(origin: .zero, size: size))
        }
    }
}

struct ThumbnailView: View {
    let image: UIImage

    private var thumbnail: Image? {
        if let thumb = image.thumbnail(size: CGSize(width: 100, height: 100)) {
            return Image(uiImage: thumb)
        }
        return nil
    }

    var body: some View {
        if let thumbnail = thumbnail {
            thumbnail
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(width: 100, height: 100)
                .clipped()
        }
    }
}
```

## アニメーションの最適化

### アニメーションの選択

```swift
struct AnimationOptimizationView: View {
    @State private var scale: CGFloat = 1.0

    var body: some View {
        VStack(spacing: 40) {
            // ✅ シンプルなアニメーションには.easeInOut
            Circle()
                .fill(.blue)
                .scaleEffect(scale)
                .animation(.easeInOut(duration: 0.3), value: scale)

            // ✅ 物理的な動きには.spring
            Circle()
                .fill(.green)
                .scaleEffect(scale)
                .animation(.spring(response: 0.3, dampingFraction: 0.6), value: scale)

            // ❌ 不要に長いアニメーション
            // .animation(.easeInOut(duration: 5.0), value: scale)

            Button("Toggle") {
                scale = scale == 1.0 ? 1.5 : 1.0
            }
        }
    }
}
```

### matchedGeometryEffect の最適化

```swift
struct OptimizedMatchedGeometryView: View {
    @State private var isExpanded = false
    @Namespace private var animation

    var body: some View {
        VStack {
            if isExpanded {
                ExpandedCard(animation: animation, isExpanded: $isExpanded)
            } else {
                CompactCard(animation: animation, isExpanded: $isExpanded)
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: isExpanded)
    }
}

struct CompactCard: View {
    let animation: Namespace.ID
    @Binding var isExpanded: Bool

    var body: some View {
        RoundedRectangle(cornerRadius: 12)
            .fill(.blue)
            .matchedGeometryEffect(id: "card", in: animation)
            .frame(width: 100, height: 100)
            .onTapGesture {
                isExpanded = true
            }
    }
}

struct ExpandedCard: View {
    let animation: Namespace.ID
    @Binding var isExpanded: Bool

    var body: some View {
        VStack {
            RoundedRectangle(cornerRadius: 12)
                .fill(.blue)
                .matchedGeometryEffect(id: "card", in: animation)
                .frame(maxWidth: .infinity)
                .frame(height: 300)

            Button("Collapse") {
                isExpanded = false
            }
        }
    }
}
```

### 60FPS維持のテクニック

```swift
struct SmoothScrollView: View {
    let items = Array(0..<100)

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                ForEach(items, id: \.self) { item in
                    OptimizedRow(item: item)
                        .id(item)
                }
            }
        }
        .scrollIndicators(.hidden)
    }
}

struct OptimizedRow: View {
    let item: Int

    var body: some View {
        HStack {
            // 重い計算は事前に完了させる
            CachedImageView(imageName: "image_\(item)")

            VStack(alignment: .leading) {
                Text("Item \(item)")
                    .font(.headline)
                Text("Description")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding()
        .background(Color(.systemBackground))
        .drawingGroup() // レイヤーをラスタライズして描画を高速化
    }
}

struct CachedImageView: View {
    let imageName: String

    var body: some View {
        Circle()
            .fill(.blue)
            .frame(width: 50, height: 50)
    }
}
```

## メモリ管理

### 強参照サイクルの回避

```swift
// ❌ 強参照サイクル
class BadNetworkManager {
    var onComplete: (() -> Void)?

    func fetchData(completion: @escaping () -> Void) {
        self.onComplete = completion
        // self と completion が互いに参照
    }
}

// ✅ weak参照で回避
class GoodNetworkManager {
    weak var delegate: NetworkDelegate?

    func fetchData() {
        // 処理
        delegate?.didFinishFetching()
    }
}

protocol NetworkDelegate: AnyObject {
    func didFinishFetching()
}

// ✅ ViewModelでのクロージャ
class ViewModel: ObservableObject {
    @Published var data: [String] = []

    func loadData() {
        NetworkService.shared.fetch { [weak self] result in
            self?.data = result
        }
    }
}

class NetworkService {
    static let shared = NetworkService()

    func fetch(completion: @escaping ([String]) -> Void) {
        // 非同期処理
    }
}
```

### メモリリークの検出

```swift
class MonitoredViewModel: ObservableObject {
    @Published var items: [Item] = []

    deinit {
        print("ViewModel deinitialized") // deinitが呼ばれることを確認
    }

    func loadItems() {
        // データロード
    }
}

struct MonitoredView: View {
    @StateObject private var viewModel = MonitoredViewModel()

    var body: some View {
        List(viewModel.items) { item in
            Text(item.title)
        }
        .onAppear {
            viewModel.loadItems()
        }
    }
}
```

### 大量データの処理

```swift
class DataProcessor: ObservableObject {
    @Published var processedData: [ProcessedItem] = []

    func processLargeDataset(_ data: [RawItem]) async {
        // バッチ処理で負荷分散
        let batchSize = 100
        var result: [ProcessedItem] = []

        for batch in data.chunked(into: batchSize) {
            let processed = await processBatch(batch)
            result.append(contentsOf: processed)

            // UIを更新して応答性を保つ
            await MainActor.run {
                self.processedData = result
            }

            // 次のバッチまで少し待つ
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
    }

    private func processBatch(_ batch: [RawItem]) async -> [ProcessedItem] {
        batch.map { ProcessedItem(from: $0) }
    }
}

struct RawItem {
    let id: Int
}

struct ProcessedItem: Identifiable {
    let id: Int

    init(from raw: RawItem) {
        self.id = raw.id
    }
}

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
```

## 非同期処理のベストプラクティス

### Task の適切な使用

```swift
struct AsyncTaskView: View {
    @StateObject private var viewModel = AsyncViewModel()

    var body: some View {
        List(viewModel.items) { item in
            Text(item.title)
        }
        .task {
            // Viewが表示されたら自動実行
            await viewModel.loadItems()
        }
        .refreshable {
            // Pull to refreshに対応
            await viewModel.refresh()
        }
    }
}

class AsyncViewModel: ObservableObject {
    @Published var items: [Item] = []
    private var loadTask: Task<Void, Never>?

    func loadItems() async {
        // 既存のタスクをキャンセル
        loadTask?.cancel()

        loadTask = Task {
            do {
                let fetchedItems = try await fetchFromAPI()

                // Task がキャンセルされていないか確認
                guard !Task.isCancelled else { return }

                await MainActor.run {
                    self.items = fetchedItems
                }
            } catch {
                print("Error: \(error)")
            }
        }
    }

    func refresh() async {
        items.removeAll()
        await loadItems()
    }

    private func fetchFromAPI() async throws -> [Item] {
        try await Task.sleep(nanoseconds: 1_000_000_000)
        return Array(0..<20).map { Item(id: UUID(), title: "Item \($0)") }
    }

    deinit {
        loadTask?.cancel()
    }
}
```

### @MainActor の活用

```swift
// ✅ ViewModelをメインアクターに閉じ込める
@MainActor
class MainActorViewModel: ObservableObject {
    @Published var items: [Item] = []

    func loadItems() async {
        // 自動的にメインスレッドで実行される
        let items = await fetchItems()
        self.items = items
    }

    private func fetchItems() async -> [Item] {
        // バックグラウンドで実行
        try? await Task.sleep(nanoseconds: 500_000_000)
        return Array(0..<10).map { Item(id: UUID(), title: "Item \($0)") }
    }
}

// ❌ 手動でMainActorに切り替え（冗長）
class ManualMainActorViewModel: ObservableObject {
    @Published var items: [Item] = []

    func loadItems() async {
        let items = await fetchItems()
        await MainActor.run {
            self.items = items
        }
    }

    private func fetchItems() async -> [Item] {
        try? await Task.sleep(nanoseconds: 500_000_000)
        return Array(0..<10).map { Item(id: UUID(), title: "Item \($0)") }
    }
}
```

### エラーハンドリング

```swift
enum NetworkError: LocalizedError {
    case invalidURL
    case noData
    case decodingError

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid URL"
        case .noData: return "No data received"
        case .decodingError: return "Failed to decode data"
        }
    }
}

class ErrorHandlingViewModel: ObservableObject {
    @Published var items: [Item] = []
    @Published var error: NetworkError?
    @Published var isLoading = false

    func loadItems() async {
        await MainActor.run {
            isLoading = true
            error = nil
        }

        do {
            let items = try await fetchItems()
            await MainActor.run {
                self.items = items
                self.isLoading = false
            }
        } catch let error as NetworkError {
            await MainActor.run {
                self.error = error
                self.isLoading = false
            }
        } catch {
            await MainActor.run {
                self.error = .noData
                self.isLoading = false
            }
        }
    }

    private func fetchItems() async throws -> [Item] {
        try await Task.sleep(nanoseconds: 500_000_000)
        return Array(0..<10).map { Item(id: UUID(), title: "Item \($0)") }
    }
}

struct ErrorHandlingView: View {
    @StateObject private var viewModel = ErrorHandlingViewModel()

    var body: some View {
        Group {
            if viewModel.isLoading {
                ProgressView()
            } else if let error = viewModel.error {
                ContentUnavailableView(
                    "Error",
                    systemImage: "exclamationmark.triangle",
                    description: Text(error.localizedDescription)
                )
            } else {
                List(viewModel.items) { item in
                    Text(item.title)
                }
            }
        }
        .task {
            await viewModel.loadItems()
        }
    }
}
```

## プレビューの最適化

### 効率的なプレビュー

```swift
struct ContentView: View {
    let items: [Item]

    var body: some View {
        List(items) { item in
            Text(item.title)
        }
    }
}

#Preview("Empty") {
    ContentView(items: [])
}

#Preview("With Data") {
    ContentView(items: Item.mockData)
}

#Preview("Many Items") {
    ContentView(items: Array(repeating: Item.mock, count: 100))
}

extension Item {
    static let mock = Item(id: UUID(), title: "Mock Item")

    static let mockData: [Item] = [
        Item(id: UUID(), title: "Item 1"),
        Item(id: UUID(), title: "Item 2"),
        Item(id: UUID(), title: "Item 3")
    ]
}
```

### プレビュー用のモック

```swift
class MockViewModel: ObservableObject {
    @Published var items: [Item] = Item.mockData
    @Published var isLoading = false

    func loadItems() async {
        // モックではデータロード不要
    }
}

struct ViewWithViewModel: View {
    @StateObject private var viewModel: MockViewModel

    init(viewModel: MockViewModel = MockViewModel()) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        List(viewModel.items) { item in
            Text(item.title)
        }
    }
}

#Preview("Loading") {
    let viewModel = MockViewModel()
    viewModel.isLoading = true
    return ViewWithViewModel(viewModel: viewModel)
}

#Preview("Loaded") {
    ViewWithViewModel()
}
```

## デバッグとプロファイリング

### デバッグ用のModifier

```swift
extension View {
    func debugPrint(_ value: Any) -> some View {
        #if DEBUG
        print(value)
        #endif
        return self
    }

    func debugBorder(_ color: Color = .red, width: CGFloat = 1) -> some View {
        #if DEBUG
        return self.border(color, width: width)
        #else
        return self
        #endif
    }

    func debugBackground(_ color: Color = .red.opacity(0.3)) -> some View {
        #if DEBUG
        return self.background(color)
        #else
        return self
        #endif
    }
}

struct DebugView: View {
    var body: some View {
        VStack {
            Text("Hello")
                .debugBorder()
                .debugPrint("Text rendered")

            Rectangle()
                .fill(.blue)
                .frame(width: 100, height: 100)
                .debugBackground(.green.opacity(0.3))
        }
    }
}
```

### パフォーマンス測定

```swift
struct PerformanceMeasurement {
    static func measure<T>(label: String, _ block: () -> T) -> T {
        let start = CFAbsoluteTimeGetCurrent()
        let result = block()
        let duration = CFAbsoluteTimeGetCurrent() - start
        print("\(label): \(duration * 1000)ms")
        return result
    }

    static func measure<T>(label: String, _ block: () async -> T) async -> T {
        let start = CFAbsoluteTimeGetCurrent()
        let result = await block()
        let duration = CFAbsoluteTimeGetCurrent() - start
        print("\(label): \(duration * 1000)ms")
        return result
    }
}

struct MeasuredView: View {
    @State private var items: [Item] = []

    var body: some View {
        List(items) { item in
            Text(item.title)
        }
        .task {
            items = await PerformanceMeasurement.measure(label: "Load Items") {
                await loadItems()
            }
        }
    }

    func loadItems() async -> [Item] {
        try? await Task.sleep(nanoseconds: 500_000_000)
        return Array(0..<100).map { Item(id: UUID(), title: "Item \($0)") }
    }
}
```

## アクセシビリティベストプラクティス

### 基本的なアクセシビリティ

```swift
struct AccessibleView: View {
    @State private var isOn = false

    var body: some View {
        VStack(spacing: 20) {
            // ✅ 明確なラベル
            Button(action: { isOn.toggle() }) {
                Image(systemName: isOn ? "star.fill" : "star")
            }
            .accessibilityLabel("Favorite")
            .accessibilityHint("Double tap to toggle favorite")

            // ✅ 状態の説明
            Toggle("Notifications", isOn: $isOn)
                .accessibilityValue(isOn ? "On" : "Off")

            // ✅ グループ化
            VStack {
                Text("John Doe")
                Text("john@example.com")
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Contact: John Doe, email: john@example.com")
        }
    }
}
```

### Dynamic Type 対応

```swift
struct DynamicTypeView: View {
    @Environment(\.dynamicTypeSize) var dynamicTypeSize

    var body: some View {
        VStack {
            Text("Title")
                .font(.title)
                .lineLimit(dynamicTypeSize.isAccessibilitySize ? nil : 2)

            Text("Body text that adapts to user's preferred text size.")
                .font(.body)

            if dynamicTypeSize < .accessibility1 {
                Image(systemName: "star")
                    .font(.largeTitle)
            }
        }
    }
}
```

## コードの可読性と保守性

### View の分割

```swift
// ❌ 大きすぎるView
struct BadLargeView: View {
    var body: some View {
        ScrollView {
            VStack {
                // ヘッダー (50行)
                HStack {
                    // ...
                }

                // メインコンテンツ (100行)
                VStack {
                    // ...
                }

                // フッター (30行)
                HStack {
                    // ...
                }
            }
        }
    }
}

// ✅ 適切に分割されたView
struct GoodModularView: View {
    var body: some View {
        ScrollView {
            VStack {
                HeaderView()
                MainContentView()
                FooterView()
            }
        }
    }
}

struct HeaderView: View {
    var body: some View {
        HStack {
            // ヘッダーのコンテンツ
        }
    }
}

struct MainContentView: View {
    var body: some View {
        VStack {
            // メインコンテンツ
        }
    }
}

struct FooterView: View {
    var body: some View {
        HStack {
            // フッターのコンテンツ
        }
    }
}
```

### View Modifier の再利用

```swift
// ✅ カスタムModifier
struct CardStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding()
            .background(.white)
            .cornerRadius(12)
            .shadow(color: .black.opacity(0.1), radius: 5, x: 0, y: 2)
    }
}

extension View {
    func cardStyle() -> some View {
        modifier(CardStyle())
    }
}

struct CardExampleView: View {
    var body: some View {
        VStack(spacing: 16) {
            Text("Card 1")
                .cardStyle()

            Text("Card 2")
                .cardStyle()

            Text("Card 3")
                .cardStyle()
        }
        .padding()
    }
}
```

### 命名規則

```swift
// ✅ 明確な命名
struct UserProfileView: View {
    let user: User
    @State private var isEditing = false
    @State private var showingDeleteAlert = false

    var body: some View {
        VStack {
            profileHeader
            profileDetails
            actionButtons
        }
    }

    private var profileHeader: some View {
        HStack {
            profileImage
            nameAndEmail
        }
    }

    private var profileImage: some View {
        Circle()
            .fill(.blue)
            .frame(width: 60, height: 60)
    }

    private var nameAndEmail: some View {
        VStack(alignment: .leading) {
            Text(user.name)
                .font(.headline)
            Text(user.email)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }

    private var profileDetails: some View {
        VStack {
            // 詳細情報
        }
    }

    private var actionButtons: some View {
        HStack {
            editButton
            deleteButton
        }
    }

    private var editButton: some View {
        Button("Edit") {
            isEditing = true
        }
    }

    private var deleteButton: some View {
        Button("Delete", role: .destructive) {
            showingDeleteAlert = true
        }
    }
}

struct User {
    let name: String
    let email: String
}
```

## テストのベストプラクティス

### ViewModelのテスト

```swift
@testable import MyApp
import XCTest

final class ViewModelTests: XCTestCase {
    var viewModel: TestableViewModel!

    override func setUp() {
        super.setUp()
        viewModel = TestableViewModel()
    }

    override func tearDown() {
        viewModel = nil
        super.tearDown()
    }

    func testInitialState() {
        XCTAssertEqual(viewModel.items.count, 0)
        XCTAssertFalse(viewModel.isLoading)
    }

    func testLoadItems() async {
        await viewModel.loadItems()

        XCTAssertEqual(viewModel.items.count, 10)
        XCTAssertFalse(viewModel.isLoading)
    }

    func testLoadItemsWithError() async {
        viewModel.shouldFail = true
        await viewModel.loadItems()

        XCTAssertNotNil(viewModel.error)
        XCTAssertEqual(viewModel.items.count, 0)
    }
}

@MainActor
class TestableViewModel: ObservableObject {
    @Published var items: [Item] = []
    @Published var isLoading = false
    @Published var error: Error?
    var shouldFail = false

    func loadItems() async {
        isLoading = true
        defer { isLoading = false }

        if shouldFail {
            error = NSError(domain: "Test", code: 1)
            return
        }

        try? await Task.sleep(nanoseconds: 100_000_000)
        items = Array(0..<10).map { Item(id: UUID(), title: "Item \($0)") }
    }
}
```

## よくある問題と解決策

### 問題1: Viewの更新が遅い

```swift
// ❌ 問題
class SlowViewModel: ObservableObject {
    @Published var items: [Item] = []

    func updateItems() {
        // メインスレッドで重い処理
        items = processHeavyData()
    }

    private func processHeavyData() -> [Item] {
        // 重い処理
        return []
    }
}

// ✅ 解決策
class FastViewModel: ObservableObject {
    @Published var items: [Item] = []

    func updateItems() async {
        // バックグラウンドで処理
        let processed = await Task.detached {
            self.processHeavyData()
        }.value

        // メインスレッドでUI更新
        await MainActor.run {
            self.items = processed
        }
    }

    private func processHeavyData() -> [Item] {
        // 重い処理
        return []
    }
}
```

### 問題2: メモリ使用量が多い

```swift
// ✅ 解決策: ページング + リソース解放
class MemoryEfficientViewModel: ObservableObject {
    @Published var visibleItems: [Item] = []
    private var allItems: [Item] = []
    private let pageSize = 50

    func loadInitialData() {
        allItems = generateLargeDataset()
        loadNextPage()
    }

    func loadNextPage() {
        let start = visibleItems.count
        let end = min(start + pageSize, allItems.count)
        visibleItems.append(contentsOf: allItems[start..<end])
    }

    func clearOldData() {
        // 表示されていない古いデータを削除
        if visibleItems.count > pageSize * 3 {
            visibleItems.removeFirst(pageSize)
        }
    }

    private func generateLargeDataset() -> [Item] {
        Array(0..<10000).map { Item(id: UUID(), title: "Item \($0)") }
    }
}
```

### 問題3: アニメーションがカクつく

```swift
// ✅ 解決策: drawingGroup() + 最適化
struct SmoothAnimationView: View {
    @State private var offset: CGFloat = 0

    var body: some View {
        ScrollView {
            LazyVStack {
                ForEach(0..<100, id: \.self) { index in
                    SimpleRow(index: index)
                }
            }
        }
        .drawingGroup() // レイヤーをラスタライズ
    }
}

struct SimpleRow: View {
    let index: Int

    var body: some View {
        HStack {
            Circle()
                .fill(.blue)
                .frame(width: 40, height: 40)
            Text("Item \(index)")
            Spacer()
        }
        .padding()
    }
}
```

---

**関連ガイド:**
- [01-state-management.md](./01-state-management.md) - 状態管理パターン
- [02-layout-navigation.md](./02-layout-navigation.md) - レイアウトとナビゲーション

**関連Skills:**
- [ios-development](../../ios-development/SKILL.md) - iOS開発全般
- [testing-strategy](../../testing-strategy/SKILL.md) - テスト戦略
- [frontend-performance](../../frontend-performance/SKILL.md) - パフォーマンス最適化

**参考資料:**
- [Optimizing SwiftUI Performance](https://developer.apple.com/documentation/swiftui/optimizing-swiftui-performance)
- [WWDC - SwiftUI Performance](https://developer.apple.com/videos/swiftui/)

**更新履歴:**
- 2025-12-30: 初版作成

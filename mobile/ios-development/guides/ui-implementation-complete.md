# iOS UI Implementation 完全ガイド

## 目次
1. [UIの基礎](#uiの基礎)
2. [SwiftUI実装](#swiftui実装)
3. [UIKit実装](#uikit実装)
4. [レイアウトテクニック](#レイアウトテクニック)
5. [カスタムビュー](#カスタムビュー)
6. [アニメーション](#アニメーション)
7. [アクセシビリティ](#アクセシビリティ)
8. [パフォーマンス最適化](#パフォーマンス最適化)

---

## UIの基礎

### SwiftUI vs UIKit

```swift
/*
┌──────────────┬────────────────┬────────────────────┐
│ 特性         │ SwiftUI        │ UIKit              │
├──────────────┼────────────────┼────────────────────┤
│ 宣言的/命令的│ 宣言的         │ 命令的             │
│ プレビュー   │ あり           │ なし（追加設定必要）│
│ 学習曲線     │ 低〜中         │ 中〜高             │
│ iOS対応      │ iOS 13+        │ すべて             │
│ カスタマイズ │ 中             │ 高                 │
│ パフォーマンス│ 高（最適化済）│ 高（手動最適化）   │
└──────────────┴────────────────┴────────────────────┘
*/

// SwiftUI - 宣言的UI
struct ProfileView: View {
    let user: User

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(user.name)
                .font(.title)
                .foregroundColor(.primary)

            Text(user.email)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Button("Edit Profile") {
                // アクション
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

// UIKit - 命令的UI
class ProfileViewController: UIViewController {
    private let user: User

    private let nameLabel = UILabel()
    private let emailLabel = UILabel()
    private let editButton = UIButton(type: .system)

    init(user: User) {
        self.user = user
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        configureViews()
        setupLayout()
    }

    private func setupUI() {
        view.backgroundColor = .systemBackground

        nameLabel.font = .preferredFont(forTextStyle: .title1)
        nameLabel.textColor = .label

        emailLabel.font = .preferredFont(forTextStyle: .subheadline)
        emailLabel.textColor = .secondaryLabel

        editButton.setTitle("Edit Profile", for: .normal)
        editButton.addTarget(self, action: #selector(editTapped), for: .touchUpInside)
    }

    private func configureViews() {
        nameLabel.text = user.name
        emailLabel.text = user.email
    }

    private func setupLayout() {
        let stackView = UIStackView(arrangedSubviews: [nameLabel, emailLabel, editButton])
        stackView.axis = .vertical
        stackView.spacing = 16
        stackView.alignment = .leading

        view.addSubview(stackView)
        stackView.translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            stackView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            stackView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            stackView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16)
        ])
    }

    @objc private func editTapped() {
        // アクション
    }
}
```

---

## SwiftUI実装

### ビュー構成とコンポーネント

```swift
// 再利用可能なコンポーネント
struct CardView<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(12)
            .shadow(color: Color.black.opacity(0.1), radius: 8, x: 0, y: 4)
    }
}

// カスタムボタンスタイル
struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding()
            .background(
                configuration.isPressed
                    ? Color.blue.opacity(0.8)
                    : Color.blue
            )
            .cornerRadius(10)
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .animation(.easeInOut(duration: 0.2), value: configuration.isPressed)
    }
}

// View Modifier
struct LoadingModifier: ViewModifier {
    let isLoading: Bool

    func body(content: Content) -> some View {
        ZStack {
            content
                .disabled(isLoading)
                .blur(radius: isLoading ? 2 : 0)

            if isLoading {
                ProgressView()
                    .scaleEffect(1.5)
            }
        }
    }
}

extension View {
    func loading(_ isLoading: Bool) -> some View {
        modifier(LoadingModifier(isLoading: isLoading))
    }
}

// 使用例
struct UserListView: View {
    @StateObject private var viewModel = UserListViewModel()

    var body: some View {
        NavigationView {
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(viewModel.users) { user in
                        CardView {
                            UserRow(user: user)
                        }
                        .onTapGesture {
                            viewModel.selectUser(user)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Users")
            .loading(viewModel.isLoading)
        }
    }
}

struct UserRow: View {
    let user: User

    var body: some View {
        HStack(spacing: 12) {
            AsyncImage(url: user.avatarURL) { image in
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } placeholder: {
                Color.gray
            }
            .frame(width: 50, height: 50)
            .clipShape(Circle())

            VStack(alignment: .leading, spacing: 4) {
                Text(user.name)
                    .font(.headline)

                Text(user.email)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Image(systemName: "chevron.right")
                .foregroundColor(.secondary)
        }
    }
}
```

### State管理

```swift
// @State - ローカル状態
struct CounterView: View {
    @State private var count = 0

    var body: some View {
        VStack {
            Text("Count: \(count)")
                .font(.largeTitle)

            Button("Increment") {
                count += 1
            }
            .buttonStyle(PrimaryButtonStyle())
        }
    }
}

// @StateObject - ViewModel
@MainActor
class FormViewModel: ObservableObject {
    @Published var name = ""
    @Published var email = ""
    @Published var isValidating = false
    @Published var errors: [String] = []

    func validate() async {
        isValidating = true
        errors.removeAll()

        // バリデーション
        if name.isEmpty {
            errors.append("Name is required")
        }

        if !email.contains("@") {
            errors.append("Invalid email")
        }

        isValidating = false
    }

    func submit() async {
        await validate()

        guard errors.isEmpty else { return }

        // Submit logic
    }
}

struct FormView: View {
    @StateObject private var viewModel = FormViewModel()

    var body: some View {
        Form {
            Section("Personal Information") {
                TextField("Name", text: $viewModel.name)
                TextField("Email", text: $viewModel.email)
                    .keyboardType(.emailAddress)
                    .textContentType(.emailAddress)
                    .autocapitalization(.none)
            }

            if !viewModel.errors.isEmpty {
                Section("Errors") {
                    ForEach(viewModel.errors, id: \.self) { error in
                        Text(error)
                            .foregroundColor(.red)
                    }
                }
            }

            Section {
                Button("Submit") {
                    Task {
                        await viewModel.submit()
                    }
                }
                .disabled(viewModel.isValidating)
            }
        }
        .loading(viewModel.isValidating)
    }
}

// @Environment - 環境値
struct ThemeKey: EnvironmentKey {
    static let defaultValue = Theme.light
}

extension EnvironmentValues {
    var theme: Theme {
        get { self[ThemeKey.self] }
        set { self[ThemeKey.self] = newValue }
    }
}

struct Theme {
    let backgroundColor: Color
    let textColor: Color

    static let light = Theme(
        backgroundColor: .white,
        textColor: .black
    )

    static let dark = Theme(
        backgroundColor: .black,
        textColor: .white
    )
}

struct ThemedView: View {
    @Environment(\.theme) private var theme

    var body: some View {
        Text("Hello, World!")
            .foregroundColor(theme.textColor)
            .background(theme.backgroundColor)
    }
}

// 使用例
struct ContentView: View {
    @State private var isDarkMode = false

    var body: some View {
        ThemedView()
            .environment(\.theme, isDarkMode ? .dark : .light)
    }
}
```

### リスト最適化

```swift
// ❌ 悪い例: パフォーマンスが低い
struct BadListView: View {
    let items: [Item]

    var body: some View {
        ScrollView {
            VStack {
                ForEach(items) { item in
                    ItemRow(item: item) // すべて一度に描画
                }
            }
        }
    }
}

// ✅ 良い例: LazyVStackで遅延読み込み
struct GoodListView: View {
    let items: [Item]

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                ForEach(items) { item in
                    ItemRow(item: item)
                        .id(item.id) // 明示的なID
                }
            }
        }
    }
}

// List with Section
struct GroupedListView: View {
    let groupedItems: [String: [Item]]

    var body: some View {
        List {
            ForEach(groupedItems.keys.sorted(), id: \.self) { key in
                Section(key) {
                    ForEach(groupedItems[key] ?? []) { item in
                        ItemRow(item: item)
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
    }
}

// Pull to Refresh
struct RefreshableListView: View {
    @StateObject private var viewModel = ListViewModel()

    var body: some View {
        List(viewModel.items) { item in
            ItemRow(item: item)
        }
        .refreshable {
            await viewModel.refresh()
        }
    }
}

// Pagination
struct PaginatedListView: View {
    @StateObject private var viewModel = PaginatedViewModel()

    var body: some View {
        List {
            ForEach(viewModel.items) { item in
                ItemRow(item: item)
                    .onAppear {
                        if item == viewModel.items.last {
                            Task {
                                await viewModel.loadMore()
                            }
                        }
                    }
            }

            if viewModel.isLoadingMore {
                ProgressView()
                    .frame(maxWidth: .infinity)
            }
        }
    }
}
```

---

## UIKit実装

### Auto Layout

```swift
// ❌ 悪い例: Frame-based layout
class BadViewController: UIViewController {
    private let label = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        label.frame = CGRect(x: 20, y: 100, width: 200, height: 40) // 回転で崩れる
        view.addSubview(label)
    }
}

// ✅ 良い例: Auto Layout
class GoodViewController: UIViewController {
    private let label = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }

    private func setupUI() {
        view.addSubview(label)
        label.translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            label.topAnchor.constraint(
                equalTo: view.safeAreaLayoutGuide.topAnchor,
                constant: 20
            ),
            label.leadingAnchor.constraint(
                equalTo: view.leadingAnchor,
                constant: 20
            ),
            label.trailingAnchor.constraint(
                equalTo: view.trailingAnchor,
                constant: -20
            )
        ])
    }
}

// Layout Anchor Extensions
extension UIView {
    func pinToSuperview(insets: UIEdgeInsets = .zero) {
        guard let superview = superview else { return }

        translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            topAnchor.constraint(equalTo: superview.topAnchor, constant: insets.top),
            leadingAnchor.constraint(equalTo: superview.leadingAnchor, constant: insets.left),
            trailingAnchor.constraint(equalTo: superview.trailingAnchor, constant: -insets.right),
            bottomAnchor.constraint(equalTo: superview.bottomAnchor, constant: -insets.bottom)
        ])
    }

    func center(in view: UIView) {
        translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            centerXAnchor.constraint(equalTo: view.centerXAnchor),
            centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])
    }
}

// Stack View
class ProfileCardView: UIView {
    private let avatarImageView = UIImageView()
    private let nameLabel = UILabel()
    private let emailLabel = UILabel()
    private let actionButton = UIButton(type: .system)

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupUI()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setupUI() {
        // Avatar
        avatarImageView.contentMode = .scaleAspectFill
        avatarImageView.layer.cornerRadius = 40
        avatarImageView.clipsToBounds = true

        // Labels
        nameLabel.font = .preferredFont(forTextStyle: .headline)
        emailLabel.font = .preferredFont(forTextStyle: .subheadline)
        emailLabel.textColor = .secondaryLabel

        // Stack Views
        let labelStack = UIStackView(arrangedSubviews: [nameLabel, emailLabel])
        labelStack.axis = .vertical
        labelStack.spacing = 4

        let contentStack = UIStackView(arrangedSubviews: [avatarImageView, labelStack])
        contentStack.axis = .horizontal
        contentStack.spacing = 12
        contentStack.alignment = .center

        let mainStack = UIStackView(arrangedSubviews: [contentStack, actionButton])
        mainStack.axis = .vertical
        mainStack.spacing = 16

        addSubview(mainStack)
        mainStack.pinToSuperview(insets: UIEdgeInsets(top: 16, left: 16, bottom: 16, right: 16))

        // Constraints for avatar
        NSLayoutConstraint.activate([
            avatarImageView.widthAnchor.constraint(equalToConstant: 80),
            avatarImageView.heightAnchor.constraint(equalToConstant: 80)
        ])
    }
}
```

### UICollectionView (Compositional Layout)

```swift
class ModernCollectionViewController: UIViewController {
    private lazy var collectionView: UICollectionView = {
        let layout = createLayout()
        let cv = UICollectionView(frame: .zero, collectionViewLayout: layout)
        cv.register(ItemCell.self, forCellWithReuseIdentifier: "ItemCell")
        cv.register(HeaderView.self, forSupplementaryViewOfKind: UICollectionView.elementKindSectionHeader, withReuseIdentifier: "Header")
        return cv
    }()

    private var dataSource: UICollectionViewDiffableDataSource<Section, Item>!

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCollectionView()
        configureDataSource()
        applySnapshot()
    }

    private func createLayout() -> UICollectionViewLayout {
        let config = UICollectionViewCompositionalLayoutConfiguration()
        config.interSectionSpacing = 20

        return UICollectionViewCompositionalLayout(sectionProvider: { sectionIndex, environment in
            let section = Section.allCases[sectionIndex]

            switch section {
            case .featured:
                return self.createFeaturedSection()
            case .grid:
                return self.createGridSection()
            case .list:
                return self.createListSection()
            }
        }, configuration: config)
    }

    private func createFeaturedSection() -> NSCollectionLayoutSection {
        // Item
        let itemSize = NSCollectionLayoutSize(
            widthDimension: .fractionalWidth(1.0),
            heightDimension: .fractionalHeight(1.0)
        )
        let item = NSCollectionLayoutItem(layoutSize: itemSize)

        // Group
        let groupSize = NSCollectionLayoutSize(
            widthDimension: .fractionalWidth(0.9),
            heightDimension: .absolute(200)
        )
        let group = NSCollectionLayoutGroup.horizontal(
            layoutSize: groupSize,
            subitems: [item]
        )

        // Section
        let section = NSCollectionLayoutSection(group: group)
        section.orthogonalScrollingBehavior = .groupPagingCentered
        section.interGroupSpacing = 16
        section.contentInsets = NSDirectionalEdgeInsets(
            top: 0, leading: 16, bottom: 0, trailing: 16
        )

        // Header
        let headerSize = NSCollectionLayoutSize(
            widthDimension: .fractionalWidth(1.0),
            heightDimension: .estimated(44)
        )
        let header = NSCollectionLayoutBoundarySupplementaryItem(
            layoutSize: headerSize,
            elementKind: UICollectionView.elementKindSectionHeader,
            alignment: .top
        )
        section.boundarySupplementaryItems = [header]

        return section
    }

    private func createGridSection() -> NSCollectionLayoutSection {
        let itemSize = NSCollectionLayoutSize(
            widthDimension: .fractionalWidth(0.5),
            heightDimension: .fractionalHeight(1.0)
        )
        let item = NSCollectionLayoutItem(layoutSize: itemSize)
        item.contentInsets = NSDirectionalEdgeInsets(
            top: 4, leading: 4, bottom: 4, trailing: 4
        )

        let groupSize = NSCollectionLayoutSize(
            widthDimension: .fractionalWidth(1.0),
            heightDimension: .absolute(150)
        )
        let group = NSCollectionLayoutGroup.horizontal(
            layoutSize: groupSize,
            subitems: [item]
        )

        let section = NSCollectionLayoutSection(group: group)
        section.contentInsets = NSDirectionalEdgeInsets(
            top: 0, leading: 12, bottom: 0, trailing: 12
        )

        return section
    }

    private func createListSection() -> NSCollectionLayoutSection {
        let config = UICollectionLayoutListConfiguration(appearance: .insetGrouped)
        return NSCollectionLayoutSection.list(using: config, layoutEnvironment: environment)
    }

    private func setupCollectionView() {
        view.addSubview(collectionView)
        collectionView.pinToSuperview()
    }

    private func configureDataSource() {
        dataSource = UICollectionViewDiffableDataSource<Section, Item>(
            collectionView: collectionView
        ) { collectionView, indexPath, item in
            let cell = collectionView.dequeueReusableCell(
                withReuseIdentifier: "ItemCell",
                for: indexPath
            ) as! ItemCell
            cell.configure(with: item)
            return cell
        }

        dataSource.supplementaryViewProvider = { collectionView, kind, indexPath in
            let header = collectionView.dequeueReusableSupplementaryView(
                ofKind: kind,
                withReuseIdentifier: "Header",
                for: indexPath
            ) as! HeaderView

            let section = Section.allCases[indexPath.section]
            header.configure(with: section.title)

            return header
        }
    }

    private func applySnapshot() {
        var snapshot = NSDiffableDataSourceSnapshot<Section, Item>()

        Section.allCases.forEach { section in
            snapshot.appendSections([section])
            snapshot.appendItems(section.items)
        }

        dataSource.apply(snapshot, animatingDifferences: true)
    }
}

enum Section: CaseIterable {
    case featured
    case grid
    case list

    var title: String {
        switch self {
        case .featured: return "Featured"
        case .grid: return "Grid"
        case .list: return "List"
        }
    }

    var items: [Item] {
        // データ生成
        []
    }
}
```

---

## アニメーション

### SwiftUIアニメーション

```swift
// 基本アニメーション
struct AnimatedButtonView: View {
    @State private var isExpanded = false

    var body: some View {
        VStack {
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.blue)
                .frame(
                    width: isExpanded ? 300 : 100,
                    height: 60
                )
                .animation(.spring(response: 0.5, dampingFraction: 0.6), value: isExpanded)

            Button("Toggle") {
                isExpanded.toggle()
            }
        }
    }
}

// マッチドジオメトリ
struct HeroAnimationView: View {
    @State private var showDetail = false
    @Namespace private var animation

    var body: some View {
        if showDetail {
            DetailView(animation: animation, showDetail: $showDetail)
        } else {
            ThumbnailView(animation: animation, showDetail: $showDetail)
        }
    }
}

struct ThumbnailView: View {
    let animation: Namespace.ID
    @Binding var showDetail: Bool

    var body: some View {
        VStack {
            Image(systemName: "photo")
                .resizable()
                .matchedGeometryEffect(id: "image", in: animation)
                .frame(width: 100, height: 100)

            Text("Tap to expand")
                .matchedGeometryEffect(id: "title", in: animation)
        }
        .onTapGesture {
            withAnimation(.spring()) {
                showDetail = true
            }
        }
    }
}

struct DetailView: View {
    let animation: Namespace.ID
    @Binding var showDetail: Bool

    var body: some View {
        VStack {
            Image(systemName: "photo")
                .resizable()
                .matchedGeometryEffect(id: "image", in: animation)
                .frame(maxWidth: .infinity, maxHeight: 300)

            Text("Detail View")
                .matchedGeometryEffect(id: "title", in: animation)
                .font(.title)

            Spacer()
        }
        .onTapGesture {
            withAnimation(.spring()) {
                showDetail = false
            }
        }
    }
}

// カスタムトランジション
extension AnyTransition {
    static var slideAndFade: AnyTransition {
        .asymmetric(
            insertion: .move(edge: .trailing).combined(with: .opacity),
            removal: .scale.combined(with: .opacity)
        )
    }
}

struct TransitionView: View {
    @State private var show = false

    var body: some View {
        VStack {
            if show {
                Rectangle()
                    .fill(Color.blue)
                    .frame(width: 200, height: 200)
                    .transition(.slideAndFade)
            }

            Button("Toggle") {
                withAnimation {
                    show.toggle()
                }
            }
        }
    }
}
```

### UIKitアニメーション

```swift
// UIView Animation
class AnimatedViewController: UIViewController {
    private let animatedView = UIView()

    func performAnimations() {
        // 基本アニメーション
        UIView.animate(withDuration: 0.3) {
            self.animatedView.alpha = 0.5
            self.animatedView.transform = CGAffineTransform(scaleX: 1.2, y: 1.2)
        }

        // スプリングアニメーション
        UIView.animate(
            withDuration: 0.6,
            delay: 0,
            usingSpringWithDamping: 0.6,
            initialSpringVelocity: 0,
            options: [.curveEaseInOut]
        ) {
            self.animatedView.transform = .identity
        }

        // キーフレームアニメーション
        UIView.animateKeyframes(
            withDuration: 2.0,
            delay: 0,
            options: []
        ) {
            UIView.addKeyframe(withRelativeStartTime: 0, relativeDuration: 0.25) {
                self.animatedView.transform = CGAffineTransform(rotationAngle: .pi / 4)
            }

            UIView.addKeyframe(withRelativeStartTime: 0.25, relativeDuration: 0.25) {
                self.animatedView.transform = CGAffineTransform(rotationAngle: .pi / 2)
            }

            UIView.addKeyframe(withRelativeStartTime: 0.5, relativeDuration: 0.25) {
                self.animatedView.transform = CGAffineTransform(rotationAngle: .pi)
            }

            UIView.addKeyframe(withRelativeStartTime: 0.75, relativeDuration: 0.25) {
                self.animatedView.transform = .identity
            }
        }
    }
}

// Core Animation
class CAAnimationViewController: UIViewController {
    private let layer = CALayer()

    func performCAAnimations() {
        // Position Animation
        let positionAnimation = CABasicAnimation(keyPath: "position")
        positionAnimation.fromValue = layer.position
        positionAnimation.toValue = CGPoint(x: 200, y: 200)
        positionAnimation.duration = 1.0
        layer.add(positionAnimation, forKey: "position")

        // Group Animation
        let scaleAnimation = CABasicAnimation(keyPath: "transform.scale")
        scaleAnimation.toValue = 1.5

        let rotateAnimation = CABasicAnimation(keyPath: "transform.rotation")
        rotateAnimation.toValue = CGFloat.pi

        let group = CAAnimationGroup()
        group.animations = [scaleAnimation, rotateAnimation]
        group.duration = 1.0
        group.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)

        layer.add(group, forKey: "groupAnimation")
    }
}
```

このガイドでは、iOSにおけるUI実装の基礎から、SwiftUIとUIKitの両方の実装パターン、アニメーション、アクセシビリティまでを網羅しました。プロジェクトの要件に応じて適切な技術を選択し、ユーザーにとって使いやすいインターフェースを構築することが重要です。

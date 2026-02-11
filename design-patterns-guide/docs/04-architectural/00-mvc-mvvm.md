# MVC / MVVM — UI 設計パターン比較

> MVC、MVP、MVVM の 3 つの UI アーキテクチャパターンを比較し、フレームワークやプラットフォームに応じた最適な選択を行うための実践ガイド。

---

## この章で学ぶこと

1. **MVC（Model-View-Controller）** の構造と Web フレームワークでの実装
2. **MVVM（Model-View-ViewModel）** のデータバインディングとリアクティブ設計
3. **各パターンの使い分け** — フレームワーク・プラットフォーム別の最適解

---

## 1. UI アーキテクチャパターンの全体像

### 1.1 3 パターンの関係

```
┌──────────────────────────────────────────────────────────┐
│              UI アーキテクチャパターンの進化               │
│                                                          │
│  1979年 Smalltalk                                        │
│  ┌────────┐                                              │
│  │  MVC   │ ← オリジナル                                 │
│  └────┬───┘                                              │
│       │ 派生                                             │
│       ├────────────────────────┐                         │
│       │                        │                         │
│  ┌────▼───┐              ┌────▼────┐                     │
│  │  MVP   │ (1990s)      │  MVVM   │ (2005 WPF)         │
│  │Presenter│              │ViewModel│                    │
│  └────────┘              └─────────┘                     │
│                                                          │
│  現代のフレームワーク:                                    │
│  Rails/Django → MVC                                     │
│  React/Vue   → MVVM に近い (Component-Based)            │
│  SwiftUI     → MVVM                                     │
│  Android     → MVVM (Jetpack)                           │
└──────────────────────────────────────────────────────────┘
```

### 1.2 3 パターンのデータフロー比較

```
┌─────────────── MVC ───────────────┐
│                                   │
│   User Action                     │
│       │                           │
│       ▼                           │
│  ┌──────────┐    更新    ┌──────┐ │
│  │Controller│ ─────────→│Model │ │
│  └──────────┘           └──┬───┘ │
│       │                    │     │
│       │ 選択View     通知  │     │
│       ▼                    ▼     │
│  ┌──────────────────────────┐    │
│  │         View             │    │
│  └──────────────────────────┘    │
└───────────────────────────────────┘

┌─────────────── MVP ───────────────┐
│                                   │
│   User Action                     │
│       │                           │
│       ▼                           │
│  ┌──────┐  イベント  ┌─────────┐  │
│  │ View │ ────────→ │Presenter│  │
│  └──────┘           └────┬────┘  │
│       ▲                  │       │
│       │ UI更新     Model操作     │
│       │                  ▼       │
│       │             ┌──────┐    │
│       └──────────── │Model │    │
│                     └──────┘    │
└───────────────────────────────────┘

┌─────────────── MVVM ──────────────┐
│                                   │
│   User Action                     │
│       │                           │
│       ▼                           │
│  ┌──────┐ Data     ┌──────────┐  │
│  │ View │←Binding→ │ViewModel │  │
│  └──────┘          └─────┬────┘  │
│                          │       │
│                    Model操作     │
│                          ▼       │
│                    ┌──────┐      │
│                    │Model │      │
│                    └──────┘      │
└───────────────────────────────────┘
```

---

## 2. MVC の実装

### 2.1 MVC の構造（Ruby on Rails の例）

```ruby
# Model — ビジネスロジックとデータアクセス
# app/models/user.rb
class User < ApplicationRecord
  has_many :posts
  validates :email, presence: true, uniqueness: true
  validates :name, presence: true, length: { maximum: 100 }

  scope :active, -> { where(active: true) }
  scope :recent, -> { order(created_at: :desc).limit(10) }

  def full_name
    "#{first_name} #{last_name}"
  end
end
```

```ruby
# Controller — リクエスト処理とレスポンス制御
# app/controllers/users_controller.rb
class UsersController < ApplicationController
  def index
    @users = User.active.recent
  end

  def show
    @user = User.find(params[:id])
  end

  def create
    @user = User.new(user_params)
    if @user.save
      redirect_to @user, notice: "ユーザーを作成しました"
    else
      render :new, status: :unprocessable_entity
    end
  end

  private

  def user_params
    params.require(:user).permit(:name, :email)
  end
end
```

```erb
<!-- View — プレゼンテーション -->
<!-- app/views/users/index.html.erb -->
<h1>ユーザー一覧</h1>
<% @users.each do |user| %>
  <div class="user-card">
    <h2><%= user.full_name %></h2>
    <p><%= user.email %></p>
    <%= link_to "詳細", user_path(user) %>
  </div>
<% end %>
```

### 2.2 MVC（Express + TypeScript）

```typescript
// Model
interface User {
  id: string;
  name: string;
  email: string;
}

class UserModel {
  async findAll(): Promise<User[]> {
    return db.query("SELECT * FROM users ORDER BY created_at DESC");
  }

  async findById(id: string): Promise<User | null> {
    return db.query("SELECT * FROM users WHERE id = $1", [id]);
  }

  async create(data: Omit<User, "id">): Promise<User> {
    return db.query(
      "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *",
      [data.name, data.email]
    );
  }
}

// Controller
class UserController {
  constructor(private model: UserModel) {}

  async index(req: Request, res: Response) {
    const users = await this.model.findAll();
    res.render("users/index", { users });
  }

  async show(req: Request, res: Response) {
    const user = await this.model.findById(req.params.id);
    if (!user) return res.status(404).render("errors/404");
    res.render("users/show", { user });
  }

  async create(req: Request, res: Response) {
    try {
      const user = await this.model.create(req.body);
      res.redirect(`/users/${user.id}`);
    } catch (error) {
      res.status(422).render("users/new", { errors: [error.message] });
    }
  }
}

// Router (View のルーティング)
const userController = new UserController(new UserModel());
router.get("/users", (req, res) => userController.index(req, res));
router.get("/users/:id", (req, res) => userController.show(req, res));
router.post("/users", (req, res) => userController.create(req, res));
```

---

## 3. MVVM の実装

### 3.1 MVVM の構造（React + hooks）

```
┌──────────────────────────────────────────────────────┐
│  React での MVVM マッピング                           │
│                                                      │
│  Model      = API クライアント + ドメインロジック     │
│  ViewModel  = Custom Hooks (useState, useEffect)     │
│  View       = JSX コンポーネント                      │
│                                                      │
│  ┌────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │ JSX (View) │←──→│ useUsers()   │───→│ API /    │  │
│  │            │    │ (ViewModel)  │    │ Domain   │  │
│  │ データ表示  │    │ 状態管理     │    │ (Model)  │  │
│  │ イベント発火│    │ ロジック     │    │          │  │
│  └────────────┘    └──────────────┘    └──────────┘  │
└──────────────────────────────────────────────────────┘
```

```typescript
// Model — API クライアントとドメインロジック
interface User {
  id: string;
  name: string;
  email: string;
  createdAt: string;
}

const userApi = {
  async fetchAll(): Promise<User[]> {
    const res = await fetch("/api/users");
    if (!res.ok) throw new Error("Failed to fetch users");
    return res.json();
  },

  async create(data: Omit<User, "id" | "createdAt">): Promise<User> {
    const res = await fetch("/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to create user");
    return res.json();
  },
};

// ViewModel — Custom Hook
function useUsers() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    loadUsers();
  }, []);

  async function loadUsers() {
    setLoading(true);
    setError(null);
    try {
      const data = await userApi.fetchAll();
      setUsers(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function addUser(name: string, email: string) {
    const newUser = await userApi.create({ name, email });
    setUsers((prev) => [newUser, ...prev]);
  }

  // 算出プロパティ（ViewModel のロジック）
  const filteredUsers = useMemo(
    () => users.filter((u) =>
      u.name.toLowerCase().includes(searchQuery.toLowerCase())
    ),
    [users, searchQuery]
  );

  return { users: filteredUsers, loading, error, searchQuery, setSearchQuery, addUser, reload: loadUsers };
}

// View — 純粋な表示コンポーネント
function UserListPage() {
  const { users, loading, error, searchQuery, setSearchQuery, addUser } = useUsers();

  if (loading) return <div>読み込み中...</div>;
  if (error) return <div>エラー: {error}</div>;

  return (
    <div>
      <h1>ユーザー一覧</h1>
      <input
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder="名前で検索"
      />
      <ul>
        {users.map((user) => (
          <li key={user.id}>
            {user.name} ({user.email})
          </li>
        ))}
      </ul>
    </div>
  );
}
```

### 3.2 MVVM（SwiftUI）

```swift
// Model
struct User: Identifiable, Codable {
    let id: UUID
    var name: String
    var email: String
    let createdAt: Date
}

class UserService {
    func fetchUsers() async throws -> [User] {
        let (data, _) = try await URLSession.shared.data(from: URL(string: "https://api.example.com/users")!)
        return try JSONDecoder().decode([User].self, from: data)
    }
}

// ViewModel
@MainActor
class UserListViewModel: ObservableObject {
    @Published var users: [User] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var searchText = ""

    private let service = UserService()

    var filteredUsers: [User] {
        guard !searchText.isEmpty else { return users }
        return users.filter { $0.name.localizedCaseInsensitiveContains(searchText) }
    }

    func loadUsers() async {
        isLoading = true
        errorMessage = nil
        do {
            users = try await service.fetchUsers()
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}

// View
struct UserListView: View {
    @StateObject private var viewModel = UserListViewModel()

    var body: some View {
        NavigationStack {
            List(viewModel.filteredUsers) { user in
                VStack(alignment: .leading) {
                    Text(user.name).font(.headline)
                    Text(user.email).font(.subheadline).foregroundColor(.secondary)
                }
            }
            .searchable(text: $viewModel.searchText)
            .navigationTitle("ユーザー一覧")
            .task { await viewModel.loadUsers() }
            .overlay {
                if viewModel.isLoading {
                    ProgressView()
                }
            }
        }
    }
}
```

---

## 4. パターンの選択基準

### 4.1 プラットフォーム / フレームワーク別の推奨

```
┌──────────────────────────────────────────────────────┐
│         フレームワーク → パターン マッピング           │
│                                                      │
│  Web (サーバーサイド):                                │
│    Rails / Django / Laravel  →  MVC (組み込み)       │
│    Express / Fastify         →  MVC (手動構成)       │
│                                                      │
│  Web (クライアントサイド):                            │
│    React                     →  MVVM (hooks)         │
│    Vue.js                    →  MVVM (Composition API)│
│    Angular                   →  MVVM (Component)     │
│    Svelte                    →  MVVM (Store)         │
│                                                      │
│  モバイル:                                           │
│    SwiftUI (iOS)             →  MVVM                 │
│    Jetpack Compose (Android) →  MVVM                 │
│    Flutter                   →  MVVM (Provider/Bloc) │
│                                                      │
│  デスクトップ:                                       │
│    WPF / .NET MAUI           →  MVVM (発祥)         │
│    Electron + React          →  MVVM                 │
└──────────────────────────────────────────────────────┘
```

---

## 5. 比較表

### 5.1 MVC / MVP / MVVM パターン比較

| 観点 | MVC | MVP | MVVM |
|------|-----|-----|------|
| **View-Logic 結合** | Controller 経由 | Presenter 経由 | DataBinding |
| **View の知識** | Controller が View を選択 | Presenter が View を更新 | ViewModel は View を知らない |
| **テスト容易性** | 中（Controller テスト） | 高（Presenter テスト） | 高（ViewModel テスト） |
| **データフロー** | 三角形（M↔V, C→M, C→V） | 直線（V↔P↔M） | 直線（V↔VM↔M） |
| **状態管理** | Model に保持 | Presenter に保持 | ViewModel に保持 |
| **複雑さ** | 低 | 中 | 中〜高 |
| **主な用途** | サーバーサイド Web | Android (古い), .NET WinForms | SPA, モバイル, WPF |

### 5.2 フレームワーク実装比較

| フレームワーク | パターン | Model | ViewModel/Controller | View | バインディング |
|--------------|---------|-------|---------------------|------|--------------|
| **Rails** | MVC | ActiveRecord | Controller | ERB/Slim | なし（テンプレート） |
| **React** | MVVM 風 | API / Store | Custom Hooks | JSX | useState / useEffect |
| **Vue 3** | MVVM | API / Pinia | Composition API | Template | ref / reactive |
| **SwiftUI** | MVVM | Service 層 | ObservableObject | View struct | @Published / @Binding |
| **Angular** | MVVM | Service | Component class | Template | [(ngModel)] |

---

## 6. アンチパターン

### 6.1 Fat Controller（MVC）

```ruby
# NG: Controller にビジネスロジックを詰め込む
class OrdersController < ApplicationController
  def create
    user = User.find(params[:user_id])
    items = params[:items].map { |i| Product.find(i[:product_id]) }

    # ビジネスロジックが Controller に...
    total = items.sum(&:price)
    tax = total * 0.10
    discount = user.vip? ? total * 0.05 : 0
    final_total = total + tax - discount

    order = Order.create!(user: user, total: final_total, tax: tax)
    items.each { |item| order.order_items.create!(product: item) }

    # 通知も Controller で...
    OrderMailer.confirmation(order).deliver_later
    SlackNotifier.new_order(order)

    redirect_to order
  end
end

# OK: Service Object に分離
class OrdersController < ApplicationController
  def create
    result = CreateOrderService.call(
      user_id: params[:user_id],
      items: params[:items]
    )

    if result.success?
      redirect_to result.order
    else
      render :new, status: :unprocessable_entity
    end
  end
end
```

### 6.2 God ViewModel（MVVM）

```typescript
// NG: 1つの ViewModel に全ロジックを詰め込む
function useGodViewModel() {
  // ユーザー管理
  const [users, setUsers] = useState([]);
  // 注文管理
  const [orders, setOrders] = useState([]);
  // 通知管理
  const [notifications, setNotifications] = useState([]);
  // 設定管理
  const [settings, setSettings] = useState({});
  // ... 100行以上のロジック
}

// OK: 責務ごとに ViewModel を分割
function useUserList() { /* ユーザー一覧のみ */ }
function useOrderManagement() { /* 注文管理のみ */ }
function useNotifications() { /* 通知のみ */ }

// コンポーネントで組み合わせ
function DashboardPage() {
  const users = useUserList();
  const orders = useOrderManagement();
  const notifications = useNotifications();
  // ...
}
```

---

## 7. FAQ

### Q1. React は MVC？ MVVM？

**A.** React 自体は「View ライブラリ」であり、特定のパターンを強制しない。ただし実際の運用では、Custom Hooks が ViewModel の役割を果たし、API クライアントが Model の役割を果たすため、MVVM に近い構造になることが多い。Facebook は当初「Flux（単方向データフロー）」を提唱したが、現在の hooks ベースの開発は MVVM のデータバインディングに近い。

### Q2. サーバーサイドとクライアントサイドの MVC は同じもの？

**A.** 名前は同じだが動作が異なる。サーバーサイド MVC（Rails 等）ではリクエスト/レスポンスの単位で Controller が動作し、View は HTML テンプレート。クライアントサイドでは状態が常にメモリに存在し、リアクティブに UI が更新される。クライアントサイドの「MVC」は実質的に MVVM に近い。

### Q3. MVVM の ViewModel が肥大化したらどうする？

**A.** 3 つのアプローチがある:
1. **ViewModel の分割** — 画面の論理的なセクションごとに ViewModel を分ける
2. **UseCase / Interactor 層の導入** — ビジネスロジックを ViewModel から抽出
3. **Composable ViewModel** — 小さな ViewModel を組み合わせて大きな画面を構成

Clean Architecture の考え方を取り入れ、ViewModel はプレゼンテーションロジック（表示の整形、UI 状態管理）のみに限定する。

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| **MVC** | サーバーサイド Web のデファクト、シンプルで理解しやすい |
| **MVP** | View と Presenter の明確な分離、テスト容易性が高い |
| **MVVM** | データバインディング、SPA とモバイルのデファクト |
| **選定基準** | フレームワークの推奨パターンに従うのが最善 |
| **共通原則** | 関心の分離、薄い Controller/ViewModel、テスト可能な設計 |

---

## 次に読むべきガイド

- [01-repository-pattern.md](./01-repository-pattern.md) — データアクセス層の抽象化
- [02-event-sourcing-cqrs.md](./02-event-sourcing-cqrs.md) — イベント駆動アーキテクチャ
- Clean Architecture ガイド — レイヤード設計の発展形

---

## 参考文献

1. **Martin Fowler** — "GUI Architectures" — https://martinfowler.com/eaaDev/uiArchs.html
2. **Microsoft** — "The MVVM Pattern" — https://learn.microsoft.com/en-us/dotnet/architecture/maui/mvvm
3. **Apple Developer Documentation** — "Model-View-ViewModel" — https://developer.apple.com/documentation/swiftui/model-data
4. **Android Developers** — "Guide to app architecture" — https://developer.android.com/topic/architecture

# MVC / MVVM — UI 設計パターン比較

> MVC、MVP、MVVM の 3 つの UI アーキテクチャパターンを比較し、フレームワークやプラットフォームに応じた最適な選択を行うための実践ガイド。各パターンの歴史的背景、データフロー、テスト戦略、アンチパターンまで、実務で必要な全知識を網羅する。

---

## 前提知識

| トピック | 必要レベル | 参照先 |
|---------|-----------|--------|
| オブジェクト指向プログラミング | 基礎（クラス、インターフェース） | [02-programming](../../../../02-programming/) |
| TypeScript / JavaScript 基礎 | 中級（型、非同期処理） | [02-programming](../../../../02-programming/) |
| HTML / CSS / DOM の基本 | 基礎 | [04-web-and-network](../../../../04-web-and-network/) |
| デザインパターンの基本概念 | 基礎（Observer、Strategy） | [../01-creational/](../01-creational/) |
| クリーンコード原則 | 基礎（関心の分離、単一責任原則） | [../../clean-code-principles/](../../clean-code-principles/) |

---

## この章で学ぶこと

1. **MVC（Model-View-Controller）** の構造と Web フレームワークでの実装、サーバーサイド・クライアントサイドでの違い
2. **MVP（Model-View-Presenter）** の特徴と MVC からの改善点、テスト容易性の向上
3. **MVVM（Model-View-ViewModel）** のデータバインディングとリアクティブ設計、現代フレームワークでの適用
4. **各パターンの使い分け** — フレームワーク・プラットフォーム別の最適解と移行戦略
5. **テスト戦略** — 各パターンにおけるユニットテスト・統合テストの書き方

---

## 1. UI アーキテクチャパターンの全体像

### WHY: なぜ UI にアーキテクチャパターンが必要か

GUI アプリケーションは「表示」「入力処理」「ビジネスロジック」「データ永続化」が密結合しやすい。パターンなしで開発すると以下の問題が発生する:

1. **テスト困難** — UI 描画なしにロジックを検証できない
2. **変更の波及** — 画面変更がビジネスロジックに影響する
3. **並行開発の阻害** — デザイナーと開発者が同一ファイルを編集する衝突
4. **コード再利用不可** — Web とモバイルで同じロジックを共有できない

これらの問題を解決するため、1979 年の Smalltalk MVC 以降、様々な UI アーキテクチャパターンが考案されてきた。

### 1.1 3 パターンの歴史的関係と進化

```
┌─────────────────────────────────────────────────────────────────────┐
│                UI アーキテクチャパターンの進化                         │
│                                                                     │
│  1979年 Smalltalk-80 (Trygve Reenskaug)                             │
│  ┌────────┐                                                         │
│  │  MVC   │ ← オリジナル: デスクトップ GUI のための分離パターン        │
│  └────┬───┘                                                         │
│       │                                                             │
│       ├─── 1996年 Taligent MVP ──────────────┐                      │
│       │    Dolphin Smalltalk MVP              │                     │
│       │                                       │                     │
│  ┌────▼────┐                            ┌────▼─────┐                │
│  │ Web MVC │ (2004 Rails)               │  MVVM    │ (2005 WPF)    │
│  │ Server  │                            │ ViewModel│ John Gossman  │
│  │ Side    │                            └──────────┘                │
│  └────┬────┘                                                        │
│       │                                                             │
│  ┌────▼────────────────────────────────────────────┐                │
│  │ Client-Side MV* (2010s)                         │                │
│  │ Backbone.js → AngularJS → React → Vue → Svelte │                │
│  │ → 実質的に MVVM + Flux/Redux ハイブリッド        │                │
│  └─────────────────────────────────────────────────┘                │
│                                                                     │
│  現代のフレームワーク:                                               │
│  Rails/Django/Laravel → Server-Side MVC                             │
│  React/Vue/Svelte    → MVVM に近い (Component-Based)               │
│  SwiftUI/Compose     → MVVM (宣言的 UI)                            │
│  Android Views       → MVP → MVVM (Jetpack)                        │
│  WPF / .NET MAUI     → MVVM (データバインディング発祥)              │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 3 パターンのデータフロー比較

```
┌────────────────── MVC ──────────────────┐
│                                         │
│   User Action                           │
│       │                                 │
│       ▼                                 │
│  ┌──────────┐    更新      ┌──────┐     │
│  │Controller│ ───────────→ │Model │     │
│  └──────────┘              └──┬───┘     │
│       │                      │          │
│       │ View選択       通知  │          │
│       ▼                      ▼          │
│  ┌────────────────────────────┐         │
│  │          View              │         │
│  └────────────────────────────┘         │
│                                         │
│  特徴: Controller が「何を表示するか」   │
│        を選択。View は Model を直接参照  │
│        することもある（Observer）        │
└─────────────────────────────────────────┘

┌────────────────── MVP ──────────────────┐
│                                         │
│   User Action                           │
│       │                                 │
│       ▼                                 │
│  ┌──────┐  イベント   ┌──────────┐      │
│  │ View │ ──────────→ │Presenter │      │
│  └──────┘             └────┬─────┘      │
│       ▲                    │            │
│       │ UI更新       Model操作          │
│       │                    ▼            │
│       │               ┌──────┐          │
│       └────────────── │Model │          │
│                       └──────┘          │
│                                         │
│  特徴: View と Presenter が 1:1 対応。   │
│        Presenter が View の参照を持ち、  │
│        明示的に UI を更新する            │
└─────────────────────────────────────────┘

┌────────────────── MVVM ─────────────────┐
│                                         │
│   User Action                           │
│       │                                 │
│       ▼                                 │
│  ┌──────┐  Data      ┌──────────┐      │
│  │ View │←─Binding──→│ViewModel │      │
│  └──────┘             └─────┬────┘      │
│                             │           │
│                       Model操作         │
│                             ▼           │
│                       ┌──────┐          │
│                       │Model │          │
│                       └──────┘          │
│                                         │
│  特徴: ViewModel は View を知らない。    │
│        データバインディングが双方向の    │
│        同期を自動処理する（リアクティブ）│
└─────────────────────────────────────────┘
```

### 1.3 各パターンの責務マトリックス

```
┌─────────────────────────────────────────────────────────────────┐
│                    責務の配置比較                                 │
│                                                                 │
│  責務              │  MVC           │  MVP          │  MVVM     │
│  ─────────────────┼────────────────┼───────────────┼───────────│
│  入力の受付       │  Controller    │  View         │  View     │
│  入力の解釈       │  Controller    │  Presenter    │  ViewModel│
│  ビジネスロジック │  Model         │  Model        │  Model    │
│  表示データ変換   │  View/Controller│  Presenter   │  ViewModel│
│  UI 描画          │  View          │  View         │  View     │
│  UI 更新トリガー  │  Model(通知)   │  Presenter    │  Binding  │
│  状態管理         │  Model         │  Presenter    │  ViewModel│
│  View の参照      │  Controller持つ │  Presenter持つ│  なし     │
│  テスト容易性     │  中            │  高           │  高       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. MVC の実装

### WHY: なぜ MVC が Web サーバーサイドのデファクトになったか

MVC は HTTP のリクエスト/レスポンスモデルと自然に対応する:
- **リクエスト** → Controller がルーティング
- **ビジネスロジック** → Model で処理
- **レスポンス** → View がHTML生成

このシンプルな対応関係が Rails (2004)、Django (2005)、Laravel (2011) などのフレームワークに採用され、Web 開発のデファクトスタンダードとなった。

### 2.1 MVC の構造（Ruby on Rails の例）

```ruby
# Model — ビジネスロジックとデータアクセス
# app/models/user.rb
class User < ApplicationRecord
  has_many :posts, dependent: :destroy
  has_many :comments
  belongs_to :organization, optional: true

  validates :email, presence: true, uniqueness: { case_sensitive: false }
  validates :name, presence: true, length: { maximum: 100 }
  validates :age, numericality: { greater_than: 0, less_than: 150 }, allow_nil: true

  scope :active, -> { where(active: true) }
  scope :recent, -> { order(created_at: :desc).limit(10) }
  scope :with_posts, -> { includes(:posts).where.not(posts: { id: nil }) }

  # ビジネスロジックは Model に置く
  def full_name
    "#{first_name} #{last_name}"
  end

  def deactivate!
    update!(active: false, deactivated_at: Time.current)
  end

  def can_post?
    active? && posts.where("created_at > ?", 1.day.ago).count < 10
  end
end
```

```ruby
# Controller — リクエスト処理とレスポンス制御
# app/controllers/users_controller.rb
class UsersController < ApplicationController
  before_action :authenticate_user!
  before_action :set_user, only: [:show, :edit, :update, :destroy]

  def index
    @users = User.active.recent
    respond_to do |format|
      format.html                    # View テンプレートを描画
      format.json { render json: @users }
    end
  end

  def show
    @posts = @user.posts.recent
  end

  def create
    @user = User.new(user_params)
    if @user.save
      UserMailer.welcome(@user).deliver_later
      redirect_to @user, notice: "ユーザーを作成しました"
    else
      render :new, status: :unprocessable_entity
    end
  end

  def update
    if @user.update(user_params)
      redirect_to @user, notice: "ユーザー情報を更新しました"
    else
      render :edit, status: :unprocessable_entity
    end
  end

  def destroy
    @user.deactivate!
    redirect_to users_path, notice: "ユーザーを無効化しました"
  end

  private

  def set_user
    @user = User.find(params[:id])
  end

  def user_params
    params.require(:user).permit(:name, :email, :first_name, :last_name, :age)
  end
end
```

```erb
<!-- View — プレゼンテーション -->
<!-- app/views/users/index.html.erb -->
<h1>ユーザー一覧</h1>

<div class="search-bar">
  <%= form_tag users_path, method: :get do %>
    <%= text_field_tag :q, params[:q], placeholder: "名前で検索..." %>
    <%= submit_tag "検索" %>
  <% end %>
</div>

<% if @users.any? %>
  <div class="user-list">
    <% @users.each do |user| %>
      <div class="user-card">
        <h2><%= user.full_name %></h2>
        <p class="email"><%= user.email %></p>
        <p class="stats">投稿数: <%= user.posts.count %></p>
        <%= link_to "詳細", user_path(user), class: "btn" %>
      </div>
    <% end %>
  </div>
  <%= paginate @users %>
<% else %>
  <p class="empty-state">ユーザーが見つかりませんでした</p>
<% end %>
```

### 2.2 MVC（Express + TypeScript）

```typescript
// ============================================================
// Model — ビジネスロジックとデータアクセス
// ============================================================
interface User {
  id: string;
  name: string;
  email: string;
  role: "admin" | "user" | "moderator";
  active: boolean;
  createdAt: Date;
}

interface CreateUserDTO {
  name: string;
  email: string;
  role?: "admin" | "user" | "moderator";
}

class UserModel {
  constructor(private db: Database) {}

  async findAll(options?: {
    page?: number;
    limit?: number;
    active?: boolean;
  }): Promise<{ users: User[]; total: number }> {
    const page = options?.page ?? 1;
    const limit = options?.limit ?? 20;
    const offset = (page - 1) * limit;

    const whereClause = options?.active !== undefined
      ? "WHERE active = $3"
      : "";

    const params = options?.active !== undefined
      ? [limit, offset, options.active]
      : [limit, offset];

    const [rows, countResult] = await Promise.all([
      this.db.query(
        `SELECT * FROM users ${whereClause} ORDER BY created_at DESC LIMIT $1 OFFSET $2`,
        params
      ),
      this.db.query(`SELECT COUNT(*) as total FROM users ${whereClause}`),
    ]);

    return { users: rows, total: countResult[0].total };
  }

  async findById(id: string): Promise<User | null> {
    const rows = await this.db.query(
      "SELECT * FROM users WHERE id = $1",
      [id]
    );
    return rows[0] ?? null;
  }

  async findByEmail(email: string): Promise<User | null> {
    const rows = await this.db.query(
      "SELECT * FROM users WHERE email = $1",
      [id]
    );
    return rows[0] ?? null;
  }

  async create(data: CreateUserDTO): Promise<User> {
    const id = crypto.randomUUID();
    const rows = await this.db.query(
      `INSERT INTO users (id, name, email, role, active, created_at)
       VALUES ($1, $2, $3, $4, true, NOW()) RETURNING *`,
      [id, data.name, data.email, data.role ?? "user"]
    );
    return rows[0];
  }

  async update(id: string, data: Partial<CreateUserDTO>): Promise<User | null> {
    const sets: string[] = [];
    const params: unknown[] = [];
    let paramIndex = 1;

    for (const [key, value] of Object.entries(data)) {
      if (value !== undefined) {
        sets.push(`${key} = $${paramIndex}`);
        params.push(value);
        paramIndex++;
      }
    }

    if (sets.length === 0) return this.findById(id);

    params.push(id);
    const rows = await this.db.query(
      `UPDATE users SET ${sets.join(", ")} WHERE id = $${paramIndex} RETURNING *`,
      params
    );
    return rows[0] ?? null;
  }
}

// ============================================================
// Controller — リクエスト処理とレスポンス制御
// ============================================================
class UserController {
  constructor(private model: UserModel) {}

  async index(req: Request, res: Response): Promise<void> {
    const page = parseInt(req.query.page as string) || 1;
    const { users, total } = await this.model.findAll({ page, active: true });

    // Content Negotiation
    if (req.accepts("json")) {
      res.json({ data: users, total, page });
    } else {
      res.render("users/index", { users, total, page });
    }
  }

  async show(req: Request, res: Response): Promise<void> {
    const user = await this.model.findById(req.params.id);
    if (!user) {
      res.status(404).render("errors/404", { message: "ユーザーが見つかりません" });
      return;
    }
    res.render("users/show", { user });
  }

  async create(req: Request, res: Response): Promise<void> {
    try {
      // バリデーション
      const { name, email, role } = req.body;
      if (!name || !email) {
        res.status(422).render("users/new", {
          errors: ["名前とメールアドレスは必須です"],
        });
        return;
      }

      // 重複チェック
      const existing = await this.model.findByEmail(email);
      if (existing) {
        res.status(422).render("users/new", {
          errors: ["このメールアドレスは既に登録されています"],
        });
        return;
      }

      const user = await this.model.create({ name, email, role });
      res.redirect(`/users/${user.id}`);
    } catch (error) {
      res.status(500).render("errors/500", {
        message: "ユーザー作成に失敗しました",
      });
    }
  }
}

// ============================================================
// Router (ルーティング定義)
// ============================================================
const userModel = new UserModel(database);
const userController = new UserController(userModel);

router.get("/users", (req, res) => userController.index(req, res));
router.get("/users/:id", (req, res) => userController.show(req, res));
router.post("/users", (req, res) => userController.create(req, res));
```

### 2.3 MVC（Django / Python）

```python
# ============================================================
# Model — Django ORM
# ============================================================
# models.py
from django.db import models
from django.core.validators import MinValueValidator

class User(models.Model):
    """ユーザーモデル — ビジネスロジックをモデルに集約"""

    class Role(models.TextChoices):
        ADMIN = "admin", "管理者"
        USER = "user", "一般ユーザー"
        MODERATOR = "moderator", "モデレーター"

    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.USER)
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.email})"

    @property
    def is_admin(self):
        return self.role == self.Role.ADMIN

    def deactivate(self):
        """ユーザーを無効化（ソフトデリート）"""
        self.active = False
        self.save(update_fields=["active"])


# ============================================================
# View (Django では View が Controller の役割)
# ============================================================
# views.py
from django.views.generic import ListView, DetailView, CreateView
from django.contrib.auth.mixins import LoginRequiredMixin

class UserListView(LoginRequiredMixin, ListView):
    """ユーザー一覧表示"""
    model = User
    template_name = "users/list.html"
    context_object_name = "users"
    paginate_by = 20

    def get_queryset(self):
        qs = super().get_queryset().filter(active=True)
        query = self.request.GET.get("q")
        if query:
            qs = qs.filter(name__icontains=query)
        return qs


class UserDetailView(LoginRequiredMixin, DetailView):
    """ユーザー詳細表示"""
    model = User
    template_name = "users/detail.html"
    context_object_name = "user"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["posts"] = self.object.posts.all()[:10]
        return context


class UserCreateView(LoginRequiredMixin, CreateView):
    """ユーザー作成"""
    model = User
    template_name = "users/form.html"
    fields = ["name", "email", "role"]
    success_url = "/users/"

    def form_valid(self, form):
        # 追加のビジネスロジック
        response = super().form_valid(form)
        send_welcome_email.delay(self.object.id)  # 非同期タスク
        return response
```

---

## 3. MVP の実装

### WHY: なぜ MVP は MVC の改良版として生まれたか

MVC の問題点は View が Model を直接参照できることにあった。これにより:
1. View と Model の結合度が高くなる
2. View のテストに Model のモックが必要
3. プレゼンテーションロジックの置き場が曖昧

MVP は Presenter を「View と Model の唯一の仲介者」として配置し、これらの問題を解決した。

### 3.1 MVP（Android Kotlin — 従来の View システム）

```kotlin
// ============================================================
// Contract — View と Presenter のインターフェースを定義
// ============================================================
interface UserListContract {
    interface View {
        fun showUsers(users: List<UserUiModel>)
        fun showLoading()
        fun hideLoading()
        fun showError(message: String)
        fun navigateToDetail(userId: String)
    }

    interface Presenter {
        fun loadUsers()
        fun onUserClicked(userId: String)
        fun onSearchQueryChanged(query: String)
        fun onDestroy()
    }
}

// ============================================================
// Model — データアクセスとビジネスロジック
// ============================================================
data class User(
    val id: String,
    val name: String,
    val email: String,
    val active: Boolean
)

data class UserUiModel(
    val id: String,
    val displayName: String,
    val email: String,
    val statusBadge: String
)

interface UserRepository {
    suspend fun getUsers(): Result<List<User>>
    suspend fun searchUsers(query: String): Result<List<User>>
}

// ============================================================
// Presenter — ロジックの中心
// ============================================================
class UserListPresenter(
    private val view: UserListContract.View,
    private val repository: UserRepository,
    private val dispatcher: CoroutineDispatcher = Dispatchers.Main
) : UserListContract.Presenter {

    private val scope = CoroutineScope(dispatcher + SupervisorJob())

    override fun loadUsers() {
        view.showLoading()
        scope.launch {
            repository.getUsers()
                .onSuccess { users ->
                    val uiModels = users.map { it.toUiModel() }
                    view.hideLoading()
                    view.showUsers(uiModels)
                }
                .onFailure { error ->
                    view.hideLoading()
                    view.showError("ユーザーの読み込みに失敗しました: ${error.message}")
                }
        }
    }

    override fun onUserClicked(userId: String) {
        view.navigateToDetail(userId)
    }

    override fun onSearchQueryChanged(query: String) {
        scope.launch {
            repository.searchUsers(query)
                .onSuccess { users ->
                    view.showUsers(users.map { it.toUiModel() })
                }
        }
    }

    override fun onDestroy() {
        scope.cancel()
    }

    private fun User.toUiModel() = UserUiModel(
        id = id,
        displayName = name,
        email = email,
        statusBadge = if (active) "Active" else "Inactive"
    )
}

// ============================================================
// View (Activity) — 描画のみ
// ============================================================
class UserListActivity : AppCompatActivity(), UserListContract.View {

    private lateinit var presenter: UserListContract.Presenter
    private lateinit var adapter: UserListAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_user_list)

        adapter = UserListAdapter { userId ->
            presenter.onUserClicked(userId)
        }
        recyclerView.adapter = adapter

        presenter = UserListPresenter(
            view = this,
            repository = UserRepositoryImpl(apiService)
        )
        presenter.loadUsers()
    }

    override fun showUsers(users: List<UserUiModel>) {
        adapter.submitList(users)
    }

    override fun showLoading() {
        progressBar.visibility = View.VISIBLE
    }

    override fun hideLoading() {
        progressBar.visibility = View.GONE
    }

    override fun showError(message: String) {
        Snackbar.make(rootView, message, Snackbar.LENGTH_LONG).show()
    }

    override fun navigateToDetail(userId: String) {
        startActivity(UserDetailActivity.intent(this, userId))
    }

    override fun onDestroy() {
        super.onDestroy()
        presenter.onDestroy()
    }
}
```

### 3.2 MVP のテスト（Presenter のユニットテスト）

```kotlin
// ============================================================
// Presenter のテスト — View のモック
// ============================================================
class UserListPresenterTest {

    private lateinit var view: UserListContract.View
    private lateinit var repository: UserRepository
    private lateinit var presenter: UserListPresenter

    @Before
    fun setup() {
        view = mock()
        repository = mock()
        // テスト用ディスパッチャーで同期実行
        presenter = UserListPresenter(view, repository, Dispatchers.Unconfined)
    }

    @Test
    fun `loadUsers - 成功時にユーザーリストを表示する`() = runTest {
        // Arrange
        val users = listOf(
            User("1", "Alice", "alice@example.com", true),
            User("2", "Bob", "bob@example.com", false),
        )
        whenever(repository.getUsers()).thenReturn(Result.success(users))

        // Act
        presenter.loadUsers()

        // Assert
        verify(view).showLoading()
        verify(view).hideLoading()
        verify(view).showUsers(argThat { size == 2 })
        verify(view, never()).showError(any())
    }

    @Test
    fun `loadUsers - 失敗時にエラーメッセージを表示する`() = runTest {
        // Arrange
        whenever(repository.getUsers()).thenReturn(
            Result.failure(IOException("Network error"))
        )

        // Act
        presenter.loadUsers()

        // Assert
        verify(view).showLoading()
        verify(view).hideLoading()
        verify(view).showError(contains("ユーザーの読み込みに失敗しました"))
        verify(view, never()).showUsers(any())
    }

    @Test
    fun `onUserClicked - 詳細画面に遷移する`() {
        // Act
        presenter.onUserClicked("user-123")

        // Assert
        verify(view).navigateToDetail("user-123")
    }
}
```

---

## 4. MVVM の実装

### WHY: なぜ MVVM が現代 UI のデファクトになったか

MVP の問題点:
1. Presenter が View の参照を持つため、ライフサイクル管理が複雑（Activity 再生成問題）
2. View のメソッドを1つずつ呼ぶ手続き的なコードが冗長
3. UI の状態が Presenter と View に分散する

MVVM はこれらを解決する:
1. **ViewModel は View を知らない** — ライフサイクル問題なし
2. **データバインディング** — 状態変更が自動的に UI に反映
3. **状態の一元管理** — ViewModel が唯一の信頼できる状態源

### 4.1 MVVM の構造（React + hooks）

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
│                                                      │
│  なぜ Custom Hook = ViewModel なのか:                │
│  1. UI とは独立したロジックの集約                     │
│  2. 状態（state）とその操作（actions）を公開          │
│  3. View を知らない（JSX への依存なし）               │
│  4. テスト可能（renderHook でテスト）                 │
└──────────────────────────────────────────────────────┘
```

```typescript
// ============================================================
// Model — API クライアントとドメインロジック
// ============================================================
interface User {
  id: string;
  name: string;
  email: string;
  role: "admin" | "user";
  createdAt: string;
}

interface CreateUserInput {
  name: string;
  email: string;
}

// API クライアント（Model 層）
class UserApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = "/api") {
    this.baseUrl = baseUrl;
  }

  async fetchAll(params?: { search?: string; page?: number }): Promise<{
    users: User[];
    total: number;
  }> {
    const query = new URLSearchParams();
    if (params?.search) query.set("q", params.search);
    if (params?.page) query.set("page", String(params.page));

    const res = await fetch(`${this.baseUrl}/users?${query}`);
    if (!res.ok) throw new ApiError("Failed to fetch users", res.status);
    return res.json();
  }

  async create(data: CreateUserInput): Promise<User> {
    const res = await fetch(`${this.baseUrl}/users`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const error = await res.json();
      throw new ApiError(error.message ?? "Failed to create user", res.status);
    }
    return res.json();
  }

  async delete(id: string): Promise<void> {
    const res = await fetch(`${this.baseUrl}/users/${id}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new ApiError("Failed to delete user", res.status);
  }
}

class ApiError extends Error {
  constructor(message: string, public statusCode: number) {
    super(message);
    this.name = "ApiError";
  }
}

// ドメインロジック（Model 層）
function sortUsers(users: User[], sortBy: "name" | "createdAt"): User[] {
  return [...users].sort((a, b) => {
    if (sortBy === "name") return a.name.localeCompare(b.name);
    return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
  });
}

function filterUsers(users: User[], query: string): User[] {
  const lower = query.toLowerCase();
  return users.filter(
    (u) =>
      u.name.toLowerCase().includes(lower) ||
      u.email.toLowerCase().includes(lower)
  );
}

// ============================================================
// ViewModel — Custom Hook
// ============================================================
interface UserListState {
  users: User[];
  total: number;
  loading: boolean;
  error: string | null;
  searchQuery: string;
  sortBy: "name" | "createdAt";
  page: number;
}

function useUserList(apiClient: UserApiClient = new UserApiClient()) {
  const [state, setState] = useState<UserListState>({
    users: [],
    total: 0,
    loading: true,
    error: null,
    searchQuery: "",
    sortBy: "createdAt",
    page: 1,
  });

  // データ取得
  const loadUsers = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const { users, total } = await apiClient.fetchAll({
        search: state.searchQuery,
        page: state.page,
      });
      setState((prev) => ({ ...prev, users, total, loading: false }));
    } catch (e) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: e instanceof Error ? e.message : "Unknown error",
      }));
    }
  }, [apiClient, state.searchQuery, state.page]);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  // アクション
  const setSearchQuery = useCallback((query: string) => {
    setState((prev) => ({ ...prev, searchQuery: query, page: 1 }));
  }, []);

  const setSortBy = useCallback((sortBy: "name" | "createdAt") => {
    setState((prev) => ({ ...prev, sortBy }));
  }, []);

  const setPage = useCallback((page: number) => {
    setState((prev) => ({ ...prev, page }));
  }, []);

  const addUser = useCallback(
    async (input: CreateUserInput) => {
      const newUser = await apiClient.create(input);
      setState((prev) => ({
        ...prev,
        users: [newUser, ...prev.users],
        total: prev.total + 1,
      }));
    },
    [apiClient]
  );

  const deleteUser = useCallback(
    async (id: string) => {
      await apiClient.delete(id);
      setState((prev) => ({
        ...prev,
        users: prev.users.filter((u) => u.id !== id),
        total: prev.total - 1,
      }));
    },
    [apiClient]
  );

  // 算出プロパティ（ViewModel のロジック）
  const sortedUsers = useMemo(
    () => sortUsers(state.users, state.sortBy),
    [state.users, state.sortBy]
  );

  const hasNextPage = state.page * 20 < state.total;
  const hasPrevPage = state.page > 1;

  return {
    // 状態
    users: sortedUsers,
    total: state.total,
    loading: state.loading,
    error: state.error,
    searchQuery: state.searchQuery,
    sortBy: state.sortBy,
    page: state.page,
    hasNextPage,
    hasPrevPage,
    // アクション
    setSearchQuery,
    setSortBy,
    setPage,
    addUser,
    deleteUser,
    reload: loadUsers,
  };
}

// ============================================================
// View — 純粋な表示コンポーネント
// ============================================================
function UserListPage() {
  const {
    users,
    loading,
    error,
    searchQuery,
    setSearchQuery,
    sortBy,
    setSortBy,
    page,
    setPage,
    hasNextPage,
    hasPrevPage,
    deleteUser,
    reload,
  } = useUserList();

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error} onRetry={reload} />;

  return (
    <div className="user-list-page">
      <h1>ユーザー一覧</h1>

      {/* 検索 + ソート */}
      <div className="controls">
        <SearchInput
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="名前 or メールで検索"
        />
        <SortSelect
          value={sortBy}
          onChange={setSortBy}
          options={[
            { value: "createdAt", label: "登録日順" },
            { value: "name", label: "名前順" },
          ]}
        />
      </div>

      {/* ユーザーリスト */}
      <ul className="user-list">
        {users.map((user) => (
          <UserCard
            key={user.id}
            user={user}
            onDelete={() => deleteUser(user.id)}
          />
        ))}
      </ul>

      {/* ページネーション */}
      <Pagination
        page={page}
        onPrev={() => setPage(page - 1)}
        onNext={() => setPage(page + 1)}
        hasPrev={hasPrevPage}
        hasNext={hasNextPage}
      />
    </div>
  );
}
```

### 4.2 MVVM（SwiftUI）

```swift
import SwiftUI
import Combine

// ============================================================
// Model — データ構造とビジネスロジック
// ============================================================
struct User: Identifiable, Codable, Equatable {
    let id: UUID
    var name: String
    var email: String
    let role: Role
    let createdAt: Date

    enum Role: String, Codable {
        case admin, user, moderator
    }

    var isAdmin: Bool { role == .admin }
}

// API クライアント（Model 層）
protocol UserServiceProtocol {
    func fetchUsers() async throws -> [User]
    func createUser(name: String, email: String) async throws -> User
    func deleteUser(id: UUID) async throws
}

class UserService: UserServiceProtocol {
    private let session: URLSession
    private let baseURL: URL

    init(baseURL: URL = URL(string: "https://api.example.com")!,
         session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
    }

    func fetchUsers() async throws -> [User] {
        let url = baseURL.appendingPathComponent("users")
        let (data, response) = try await session.data(from: url)
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw APIError.invalidResponse
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode([User].self, from: data)
    }

    func createUser(name: String, email: String) async throws -> User {
        var request = URLRequest(url: baseURL.appendingPathComponent("users"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["name": name, "email": email])

        let (data, _) = try await session.data(for: request)
        return try JSONDecoder().decode(User.self, from: data)
    }

    func deleteUser(id: UUID) async throws {
        var request = URLRequest(
            url: baseURL.appendingPathComponent("users/\(id.uuidString)")
        )
        request.httpMethod = "DELETE"
        let (_, _) = try await session.data(for: request)
    }
}

enum APIError: LocalizedError {
    case invalidResponse
    case networkError(Error)

    var errorDescription: String? {
        switch self {
        case .invalidResponse: return "サーバーからの応答が不正です"
        case .networkError(let error): return error.localizedDescription
        }
    }
}

// ============================================================
// ViewModel — UIの状態とロジック
// ============================================================
@MainActor
class UserListViewModel: ObservableObject {
    // Published = データバインディング対象
    @Published var users: [User] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var searchText = ""
    @Published var showingAddSheet = false

    private let service: UserServiceProtocol

    init(service: UserServiceProtocol = UserService()) {
        self.service = service
    }

    // 算出プロパティ — View が参照する表示用データ
    var filteredUsers: [User] {
        guard !searchText.isEmpty else { return users }
        return users.filter {
            $0.name.localizedCaseInsensitiveContains(searchText) ||
            $0.email.localizedCaseInsensitiveContains(searchText)
        }
    }

    var userCount: String {
        let count = filteredUsers.count
        return "\(count)件のユーザー"
    }

    var hasUsers: Bool { !filteredUsers.isEmpty }

    // アクション
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

    func addUser(name: String, email: String) async {
        do {
            let newUser = try await service.createUser(name: name, email: email)
            users.insert(newUser, at: 0)
        } catch {
            errorMessage = "ユーザーの追加に失敗しました: \(error.localizedDescription)"
        }
    }

    func deleteUser(at offsets: IndexSet) async {
        let usersToDelete = offsets.map { filteredUsers[$0] }
        for user in usersToDelete {
            do {
                try await service.deleteUser(id: user.id)
                users.removeAll { $0.id == user.id }
            } catch {
                errorMessage = "削除に失敗しました"
            }
        }
    }
}

// ============================================================
// View — 宣言的 UI
// ============================================================
struct UserListView: View {
    @StateObject private var viewModel = UserListViewModel()

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.isLoading {
                    ProgressView("読み込み中...")
                } else if let error = viewModel.errorMessage {
                    ContentUnavailableView(
                        "エラー",
                        systemImage: "exclamationmark.triangle",
                        description: Text(error)
                    )
                } else if viewModel.hasUsers {
                    userList
                } else {
                    ContentUnavailableView.search
                }
            }
            .navigationTitle("ユーザー一覧")
            .searchable(text: $viewModel.searchText, prompt: "名前 or メールで検索")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("追加") { viewModel.showingAddSheet = true }
                }
                ToolbarItem(placement: .status) {
                    Text(viewModel.userCount)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .sheet(isPresented: $viewModel.showingAddSheet) {
                AddUserSheet(viewModel: viewModel)
            }
            .refreshable {
                await viewModel.loadUsers()
            }
            .task {
                await viewModel.loadUsers()
            }
        }
    }

    private var userList: some View {
        List {
            ForEach(viewModel.filteredUsers) { user in
                NavigationLink(value: user) {
                    UserRow(user: user)
                }
            }
            .onDelete { offsets in
                Task { await viewModel.deleteUser(at: offsets) }
            }
        }
    }
}

struct UserRow: View {
    let user: User

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(user.name).font(.headline)
                if user.isAdmin {
                    Text("Admin")
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.blue.opacity(0.2))
                        .cornerRadius(4)
                }
            }
            Text(user.email)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }
}
```

### 4.3 MVVM（Vue 3 Composition API）

```typescript
// ============================================================
// Model — API クライアント
// ============================================================
// api/userApi.ts
import type { User, CreateUserInput } from '@/types'

export const userApi = {
  async fetchAll(params?: { search?: string; page?: number }): Promise<{
    users: User[];
    total: number;
  }> {
    const query = new URLSearchParams()
    if (params?.search) query.set('q', params.search)
    if (params?.page) query.set('page', String(params.page))
    const res = await fetch(`/api/users?${query}`)
    if (!res.ok) throw new Error('Failed to fetch users')
    return res.json()
  },

  async create(data: CreateUserInput): Promise<User> {
    const res = await fetch('/api/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    if (!res.ok) throw new Error('Failed to create user')
    return res.json()
  },
}

// ============================================================
// ViewModel — Composable
// ============================================================
// composables/useUserList.ts
import { ref, computed, watch } from 'vue'
import { userApi } from '@/api/userApi'
import type { User } from '@/types'

export function useUserList() {
  // リアクティブ状態
  const users = ref<User[]>([])
  const total = ref(0)
  const loading = ref(false)
  const error = ref<string | null>(null)
  const searchQuery = ref('')
  const sortBy = ref<'name' | 'createdAt'>('createdAt')

  // 算出プロパティ（リアクティブに自動更新）
  const sortedUsers = computed(() => {
    return [...users.value].sort((a, b) => {
      if (sortBy.value === 'name') return a.name.localeCompare(b.name)
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    })
  })

  const isEmpty = computed(() => users.value.length === 0 && !loading.value)

  // アクション
  async function loadUsers() {
    loading.value = true
    error.value = null
    try {
      const result = await userApi.fetchAll({ search: searchQuery.value })
      users.value = result.users
      total.value = result.total
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error'
    } finally {
      loading.value = false
    }
  }

  async function addUser(name: string, email: string) {
    const newUser = await userApi.create({ name, email })
    users.value.unshift(newUser)
    total.value++
  }

  // searchQuery の変更を監視して自動検索
  watch(searchQuery, () => {
    loadUsers()
  }, { debounce: 300 } as any)

  // 初期ロード
  loadUsers()

  return {
    users: sortedUsers,
    total,
    loading,
    error,
    searchQuery,
    sortBy,
    isEmpty,
    loadUsers,
    addUser,
  }
}

// ============================================================
// View — テンプレート
// ============================================================
// components/UserListPage.vue
// <script setup lang="ts">
// import { useUserList } from '@/composables/useUserList'
//
// const {
//   users, loading, error, searchQuery, sortBy, isEmpty, addUser, loadUsers
// } = useUserList()
// </script>
//
// <template>
//   <div class="user-list-page">
//     <h1>ユーザー一覧</h1>
//     <input v-model="searchQuery" placeholder="検索..." />
//     <select v-model="sortBy">
//       <option value="createdAt">登録日順</option>
//       <option value="name">名前順</option>
//     </select>
//     <div v-if="loading">読み込み中...</div>
//     <div v-else-if="error">{{ error }}</div>
//     <div v-else-if="isEmpty">ユーザーが見つかりません</div>
//     <ul v-else>
//       <li v-for="user in users" :key="user.id">
//         {{ user.name }} ({{ user.email }})
//       </li>
//     </ul>
//   </div>
// </template>
```

### 4.4 MVVM のテスト（React Custom Hook のテスト）

```typescript
// ============================================================
// ViewModel (Custom Hook) のテスト
// ============================================================
import { renderHook, act, waitFor } from "@testing-library/react";
import { useUserList } from "./useUserList";

// Mock API Client
const mockApiClient = {
  fetchAll: jest.fn(),
  create: jest.fn(),
  delete: jest.fn(),
} as unknown as UserApiClient;

describe("useUserList (ViewModel)", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("初期ロードでユーザー一覧を取得する", async () => {
    const mockUsers = [
      { id: "1", name: "Alice", email: "alice@example.com", role: "user", createdAt: "2024-01-01" },
      { id: "2", name: "Bob", email: "bob@example.com", role: "admin", createdAt: "2024-01-02" },
    ];
    (mockApiClient.fetchAll as jest.Mock).mockResolvedValue({
      users: mockUsers,
      total: 2,
    });

    const { result } = renderHook(() => useUserList(mockApiClient));

    // 初期状態: ローディング中
    expect(result.current.loading).toBe(true);

    // データ取得完了を待つ
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // 検証
    expect(result.current.users).toHaveLength(2);
    expect(result.current.total).toBe(2);
    expect(result.current.error).toBeNull();
  });

  test("検索クエリを変更するとページがリセットされる", async () => {
    (mockApiClient.fetchAll as jest.Mock).mockResolvedValue({
      users: [],
      total: 0,
    });

    const { result } = renderHook(() => useUserList(mockApiClient));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // ページを2に設定
    act(() => {
      result.current.setPage(2);
    });
    expect(result.current.page).toBe(2);

    // 検索すると page が 1 にリセットされる
    act(() => {
      result.current.setSearchQuery("Alice");
    });
    expect(result.current.page).toBe(1);
    expect(result.current.searchQuery).toBe("Alice");
  });

  test("ユーザー追加で楽観的更新される", async () => {
    const initialUsers = [
      { id: "1", name: "Alice", email: "alice@example.com", role: "user", createdAt: "2024-01-01" },
    ];
    (mockApiClient.fetchAll as jest.Mock).mockResolvedValue({
      users: initialUsers,
      total: 1,
    });

    const newUser = {
      id: "2",
      name: "Bob",
      email: "bob@example.com",
      role: "user",
      createdAt: "2024-01-02",
    };
    (mockApiClient.create as jest.Mock).mockResolvedValue(newUser);

    const { result } = renderHook(() => useUserList(mockApiClient));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // ユーザー追加
    await act(async () => {
      await result.current.addUser({ name: "Bob", email: "bob@example.com" });
    });

    // 楽観的更新の検証
    expect(result.current.users).toHaveLength(2);
    expect(result.current.total).toBe(2);
  });

  test("API エラー時にエラーメッセージが設定される", async () => {
    (mockApiClient.fetchAll as jest.Mock).mockRejectedValue(
      new Error("Network error")
    );

    const { result } = renderHook(() => useUserList(mockApiClient));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe("Network error");
    expect(result.current.users).toHaveLength(0);
  });
});
```

---

## 5. パターンの選択基準

### 5.1 プラットフォーム / フレームワーク別の推奨

```
┌──────────────────────────────────────────────────────────────┐
│          フレームワーク → パターン マッピング                  │
│                                                              │
│  Web (サーバーサイド):                                       │
│    Rails / Django / Laravel   → MVC (フレームワーク組み込み)  │
│    Express / Fastify / Hono   → MVC (手動構成、薄い Controller)│
│    Spring Boot                → MVC (@Controller, @Service)   │
│    Go (net/http, Gin, Echo)   → MVC (ハンドラ + サービス)     │
│                                                              │
│  Web (クライアントサイド):                                   │
│    React                      → MVVM (Custom Hooks = VM)     │
│    Vue.js 3                   → MVVM (Composition API = VM)  │
│    Angular                    → MVVM (Component + Service)   │
│    Svelte                     → MVVM (Store = VM)            │
│    Solid.js                   → MVVM (createSignal = VM)     │
│                                                              │
│  モバイル:                                                   │
│    SwiftUI (iOS)              → MVVM (ObservableObject)      │
│    UIKit (iOS, レガシー)       → MVC → MVP                    │
│    Jetpack Compose (Android)  → MVVM (StateFlow + ViewModel) │
│    Android Views (レガシー)    → MVP → MVVM (LiveData)        │
│    Flutter                    → MVVM (Provider / Bloc / Riverpod)│
│    React Native               → MVVM (hooks ベース)           │
│                                                              │
│  デスクトップ:                                               │
│    WPF / .NET MAUI            → MVVM (発祥、INotifyPropertyChanged)│
│    Electron + React           → MVVM                         │
│    Tauri + Solid/Svelte       → MVVM                         │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 選定フローチャート

```
┌────────────────────────────────────────────────────────────┐
│               パターン選定フローチャート                      │
│                                                            │
│  Q1: サーバーサイド Web アプリ？                             │
│    Yes → MVC（フレームワークの規約に従う）                   │
│    No  → Q2 へ                                             │
│                                                            │
│  Q2: リアクティブ UI / SPA / モバイル？                     │
│    Yes → Q3 へ                                             │
│    No  → MVC（シンプルなサーバーレンダリング）               │
│                                                            │
│  Q3: フレームワークがデータバインディングをサポート？        │
│    Yes → MVVM（React hooks, SwiftUI, Vue Composition等）   │
│    No  → Q4 へ                                             │
│                                                            │
│  Q4: View のテスト容易性を最重視？                          │
│    Yes → MVP（View インターフェース経由でテスト）            │
│    No  → MVC（カスタム実装）                                │
└────────────────────────────────────────────────────────────┘
```

---

## 6. 比較表

### 6.1 MVC / MVP / MVVM パターン比較

| 観点 | MVC | MVP | MVVM |
|------|-----|-----|------|
| **View-Logic 結合** | Controller 経由 | Presenter 経由 | DataBinding |
| **View の知識** | Controller が View を選択 | Presenter が View を更新 | ViewModel は View を知らない |
| **テスト容易性** | 中（Controller テスト） | 高（Presenter テスト） | 高（ViewModel テスト） |
| **データフロー** | 三角形（M<->V, C->M, C->V） | 直線（V<->P<->M） | 直線（V<->VM<->M） |
| **状態管理** | Model に保持 | Presenter に保持 | ViewModel に保持 |
| **View の役割** | 表示 + 一部ロジック | 表示のみ（Passive View） | 表示 + バインディング |
| **複雑さ** | 低 | 中 | 中〜高 |
| **学習コスト** | 低 | 中 | 中（リアクティブ理解必要） |
| **ボイラープレート** | 少ない | 多い（Contract 定義） | 中（バインディング設定） |
| **主な用途** | サーバーサイド Web | Android (旧), .NET WinForms | SPA, モバイル, WPF |
| **代表フレームワーク** | Rails, Django, Laravel | Android Views | React, SwiftUI, WPF |

### 6.2 フレームワーク実装比較

| フレームワーク | パターン | Model | ViewModel/Controller | View | バインディング |
|--------------|---------|-------|---------------------|------|--------------|
| **Rails** | MVC | ActiveRecord | Controller | ERB/Slim | なし（テンプレート） |
| **Django** | MVC (MTV) | ORM Model | View (=Controller) | Template | なし（テンプレート） |
| **Spring Boot** | MVC | @Entity / @Service | @Controller | Thymeleaf | なし（テンプレート） |
| **React** | MVVM 風 | API / Store | Custom Hooks | JSX | useState / useEffect |
| **Vue 3** | MVVM | API / Pinia | Composition API | Template | ref / reactive |
| **Svelte** | MVVM | API / Store | $state rune | Template | 自動リアクティブ |
| **Angular** | MVVM | Service / NgRx | Component class | Template | [(ngModel)] / Signal |
| **SwiftUI** | MVVM | Service 層 | ObservableObject | View struct | @Published / @Binding |
| **Jetpack Compose** | MVVM | Repository | ViewModel | @Composable | StateFlow / MutableState |
| **WPF** | MVVM | Data Layer | ViewModel (INotifyPropertyChanged) | XAML | {Binding} |
| **Flutter** | MVVM 風 | Repository | Provider/Bloc/Riverpod | Widget | ChangeNotifier / Stream |

### 6.3 テスト戦略比較

| パターン | ユニットテスト対象 | モック対象 | テストの書きやすさ | UI テスト必要度 |
|---------|------------------|-----------|-------------------|---------------|
| **MVC** | Model, Controller | DB, 外部 API | 中 | 高（View ロジックあり） |
| **MVP** | Model, Presenter | View (Interface), DB | 高 | 低（View はパッシブ） |
| **MVVM** | Model, ViewModel | API Client | 高 | 低（VM に全ロジック） |

---

## 7. アンチパターン

### 7.1 Fat Controller（MVC）

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
# app/services/create_order_service.rb
class CreateOrderService
  def initialize(user_id:, items:)
    @user = User.find(user_id)
    @items = items.map { |i| Product.find(i[:product_id]) }
  end

  def call
    order = build_order
    persist_order(order)
    send_notifications(order)
    Result.success(order)
  rescue => e
    Result.failure(e.message)
  end

  private

  def build_order
    calculator = PriceCalculator.new(@items, @user)
    Order.new(
      user: @user,
      total: calculator.final_total,
      tax: calculator.tax
    )
  end

  def persist_order(order)
    ActiveRecord::Base.transaction do
      order.save!
      @items.each { |item| order.order_items.create!(product: item) }
    end
  end

  def send_notifications(order)
    OrderMailer.confirmation(order).deliver_later
    SlackNotifier.new_order(order)
  end
end

# Controller は薄く
class OrdersController < ApplicationController
  def create
    result = CreateOrderService.new(
      user_id: params[:user_id],
      items: params[:items]
    ).call

    if result.success?
      redirect_to result.order
    else
      render :new, status: :unprocessable_entity
    end
  end
end
```

**なぜ NG か**: Controller はリクエストのルーティングとレスポンスの制御のみを担当すべき。ビジネスロジックを含むと、テストが困難になり、ロジックの再利用もできない。

### 7.2 God ViewModel（MVVM）

```typescript
// NG: 1つの ViewModel に全ロジックを詰め込む
function useDashboardGodViewModel() {
  // ユーザー管理
  const [users, setUsers] = useState([]);
  const [userSearch, setUserSearch] = useState("");
  // 注文管理
  const [orders, setOrders] = useState([]);
  const [orderFilter, setOrderFilter] = useState("all");
  // 通知管理
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  // 設定管理
  const [settings, setSettings] = useState({});
  const [theme, setTheme] = useState("light");
  // 分析ダッシュボード
  const [analytics, setAnalytics] = useState({});
  const [dateRange, setDateRange] = useState({ from: null, to: null });
  // ... 200行以上のロジック

  return { /* 50+ のプロパティとメソッド */ };
}

// OK: 責務ごとに ViewModel を分割
function useUserList() {
  const [users, setUsers] = useState<User[]>([]);
  const [search, setSearch] = useState("");
  // ユーザー一覧に関するロジックのみ（30行以内）
  return { users, search, setSearch, loadUsers, addUser };
}

function useOrderManagement() {
  const [orders, setOrders] = useState<Order[]>([]);
  const [filter, setFilter] = useState<OrderFilter>("all");
  // 注文管理に関するロジックのみ
  return { orders, filter, setFilter, loadOrders };
}

function useNotifications() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  // 通知に関するロジックのみ
  return { notifications, unreadCount, markAsRead };
}

// コンポーネントで必要なものだけ組み合わせ
function DashboardPage() {
  const userList = useUserList();
  const orders = useOrderManagement();
  const notifications = useNotifications();

  return (
    <Dashboard>
      <UserSection {...userList} />
      <OrderSection {...orders} />
      <NotificationBell {...notifications} />
    </Dashboard>
  );
}
```

**なぜ NG か**: ViewModel が肥大化すると、テストが困難になり、変更の影響範囲が読めなくなる。1つの ViewModel の目安は状態3-5個、メソッド5-8個以内。

### 7.3 View にビジネスロジック（共通）

```typescript
// NG: View コンポーネント内にビジネスロジック
function OrderSummary({ order }: { order: Order }) {
  // ビジネスロジックが View に混在
  const subtotal = order.items.reduce((sum, item) => sum + item.price * item.qty, 0);
  const taxRate = order.country === "JP" ? 0.10 : order.country === "US" ? 0.08 : 0.20;
  const tax = subtotal * taxRate;
  const discount = order.coupon
    ? order.coupon.type === "percent"
      ? subtotal * (order.coupon.value / 100)
      : order.coupon.value
    : 0;
  const total = subtotal + tax - discount;
  const freeShipping = total > 5000;

  return (
    <div>
      <p>小計: {subtotal.toLocaleString()}円</p>
      <p>税: {tax.toLocaleString()}円</p>
      <p>割引: -{discount.toLocaleString()}円</p>
      <p>合計: {total.toLocaleString()}円</p>
      {freeShipping && <p>送料無料</p>}
    </div>
  );
}

// OK: ViewModel にビジネスロジックを移動
function useOrderSummary(order: Order) {
  return useMemo(() => {
    const calculator = new OrderCalculator(order);
    return {
      subtotal: calculator.subtotal,
      tax: calculator.tax,
      discount: calculator.discount,
      total: calculator.total,
      freeShipping: calculator.isFreeShipping,
    };
  }, [order]);
}

// View は表示のみ
function OrderSummary({ order }: { order: Order }) {
  const { subtotal, tax, discount, total, freeShipping } = useOrderSummary(order);

  return (
    <div>
      <p>小計: {subtotal.toLocaleString()}円</p>
      <p>税: {tax.toLocaleString()}円</p>
      <p>割引: -{discount.toLocaleString()}円</p>
      <p>合計: {total.toLocaleString()}円</p>
      {freeShipping && <p>送料無料</p>}
    </div>
  );
}
```

**なぜ NG か**: View にビジネスロジックがあると、(1) 同じ計算を複数の View で重複実装する、(2) ロジックのユニットテストに UI レンダリングが必要になる、(3) デザイナーがレイアウト変更時にロジックを壊すリスクがある。

### 7.4 ViewModel から View の直接操作（MVVM 違反）

```typescript
// NG: ViewModel が DOM を直接操作
function useScrollToTop() {
  const scrollToTop = () => {
    // ViewModel が View (DOM) を知っている！
    document.getElementById("scroll-container")?.scrollTo(0, 0);
    document.title = "ページトップ";
  };
  return { scrollToTop };
}

// OK: ViewModel は状態のみを公開し、View が副作用を実行
function useListViewModel() {
  const [shouldScrollToTop, setShouldScrollToTop] = useState(false);

  const resetList = () => {
    // 状態のみを変更
    setShouldScrollToTop(true);
    setPage(1);
  };

  return { shouldScrollToTop, setShouldScrollToTop, resetList };
}

// View が副作用を実行
function ListView() {
  const { shouldScrollToTop, setShouldScrollToTop, resetList } = useListViewModel();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (shouldScrollToTop) {
      containerRef.current?.scrollTo(0, 0);
      setShouldScrollToTop(false);
    }
  }, [shouldScrollToTop]);

  return <div ref={containerRef}>...</div>;
}
```

**なぜ NG か**: MVVM の核心は「ViewModel が View を知らない」こと。ViewModel が DOM を操作すると、(1) ViewModel がブラウザ環境でしかテストできない、(2) React Native などへの移植ができない、(3) SSR でエラーになる。

---

## 8. 実践演習

### 演習 1（基礎）: MVC の責務分離

以下の「Fat Controller」を、適切に Model / Controller に責務を分離してリファクタリングせよ。

```typescript
// リファクタリング対象
class ProductController {
  async search(req: Request, res: Response) {
    const { query, minPrice, maxPrice, category } = req.query;
    const products = await db.query("SELECT * FROM products");

    // フィルタリング（ビジネスロジック）
    let filtered = products;
    if (query) {
      filtered = filtered.filter(p =>
        p.name.toLowerCase().includes(query.toLowerCase())
      );
    }
    if (minPrice) {
      filtered = filtered.filter(p => p.price >= Number(minPrice));
    }
    if (maxPrice) {
      filtered = filtered.filter(p => p.price <= Number(maxPrice));
    }
    if (category) {
      filtered = filtered.filter(p => p.category === category);
    }

    // ソート（ビジネスロジック）
    filtered.sort((a, b) => b.salesCount - a.salesCount);

    // ページネーション
    const page = Number(req.query.page) || 1;
    const perPage = 20;
    const start = (page - 1) * perPage;
    const paged = filtered.slice(start, start + perPage);

    res.json({ data: paged, total: filtered.length });
  }
}
```

**期待する出力構造**:
- `ProductModel` クラス: `search(criteria)`, `sortByPopularity()`, `paginate(page, perPage)` メソッド
- `ProductController` クラス: 薄いリクエスト処理のみ（10行以内）

---

### 演習 2（応用）: MVVM のテスト可能な設計

TODO リストアプリの ViewModel を設計せよ。以下の要件を満たすこと:

**要件**:
1. TODO の追加・完了トグル・削除
2. フィルタリング: All / Active / Completed
3. 残りの未完了タスク数の表示
4. 全完了トグル（全てのタスクを一括完了/未完了）

**テスト**: 以下のテストケースが全てパスする ViewModel を実装せよ:

```typescript
describe("useTodoList", () => {
  test("TODO を追加できる", () => {
    // addTodo("Buy milk") → todos の length が 1 増える
  });

  test("空文字の TODO は追加できない", () => {
    // addTodo("") → todos の length は変わらない
  });

  test("TODO の完了状態をトグルできる", () => {
    // toggleTodo(id) → completed が反転
  });

  test("フィルターで Active のみ表示できる", () => {
    // filter = "active" → completed: false のみ
  });

  test("残りの未完了タスク数が正しい", () => {
    // 3個中1個完了 → remainingCount = 2
  });

  test("全完了トグルが動作する", () => {
    // toggleAll() → 全て completed: true
    // もう一度 toggleAll() → 全て completed: false
  });
});
```

**期待する出力**: `useTodoList()` Custom Hook の完全な実装とテストコード

---

### 演習 3（発展）: パターン間の移行

既存の MVP 実装（Android Kotlin）を MVVM（Jetpack Compose）に移行せよ。以下の MVP コードを起点として:

```kotlin
// 既存 MVP コード
interface WeatherContract {
    interface View {
        fun showTemperature(temp: String)
        fun showCondition(condition: String)
        fun showLoading()
        fun hideLoading()
        fun showError(message: String)
    }
    interface Presenter {
        fun loadWeather(city: String)
        fun onDestroy()
    }
}
```

**移行要件**:
1. `WeatherViewModel` (extends `ViewModel()`) として再設計
2. `StateFlow` を使用した UI 状態管理
3. `sealed class WeatherUiState` で状態を型安全に表現
4. Compose UI の `@Composable` 関数として View を実装
5. `FakeWeatherRepository` を使ったユニットテスト

**期待する出力**:
- `WeatherUiState` sealed class
- `WeatherViewModel` クラス
- `WeatherScreen` Composable
- `WeatherViewModelTest` テストクラス

---

## 9. FAQ

### Q1. React は MVC？ MVVM？

**A.** React 自体は「View ライブラリ」であり、特定のパターンを強制しない。ただし実際の運用では:

- **Custom Hooks** = ViewModel（状態管理、ビジネスロジック）
- **API クライアント / Store** = Model（データアクセス、ドメインロジック）
- **JSX コンポーネント** = View（UI 描画）

この構造は **MVVM に近い**。Facebook は当初「Flux（単方向データフロー）」を提唱したが、Hooks 登場後の開発スタイルは MVVM のデータバインディングに極めて近い。`useState` と `useEffect` の組み合わせが暗黙のバインディング機構として機能している。

ただし注意点として、React の「状態更新 → 再レンダリング」は **単方向** であり、WPF のような **双方向** バインディングとは異なる。厳密には「単方向データバインディングの MVVM 変種」と言える。

### Q2. サーバーサイドとクライアントサイドの MVC は同じもの？

**A.** 名前は同じだが動作モデルが根本的に異なる:

| 比較軸 | サーバーサイド MVC | クライアントサイド MVC |
|--------|-------------------|---------------------|
| **ライフサイクル** | リクエスト/レスポンス単位（ステートレス） | アプリ起動〜終了（ステートフル） |
| **状態の保持** | DB + セッション | メモリ内（リアクティブ） |
| **View の更新** | HTML 全体を再生成 | DOM の差分更新 |
| **ユーザー操作** | HTTP リクエスト | イベントハンドラ |
| **Controller の寿命** | リクエスト処理中のみ | アプリ全体 |

サーバーサイドの Controller は「1リクエスト = 1インスタンス」で使い捨てだが、クライアントサイドの Controller/ViewModel は継続的に生存する。この違いにより、クライアントサイドでは MVC より **MVVM のほうが自然に適合する**。

### Q3. MVVM の ViewModel が肥大化したらどうする？

**A.** 3 つのアプローチがある:

1. **ViewModel の分割** — 画面の論理的なセクションごとに ViewModel を分ける
   ```typescript
   // 1つの画面に複数の ViewModel
   function Dashboard() {
     const header = useHeaderViewModel();
     const userList = useUserListViewModel();
     const stats = useStatsViewModel();
   }
   ```

2. **UseCase / Interactor 層の導入** — ビジネスロジックを ViewModel から抽出
   ```typescript
   // ViewModel は UseCase を呼ぶだけ
   function useOrderViewModel(createOrder: CreateOrderUseCase) {
     const submit = async (data) => {
       const result = await createOrder.execute(data);
       // ViewModel はプレゼンテーションロジックのみ
     };
   }
   ```

3. **Composable ViewModel** — 小さな ViewModel を組み合わせて大きな画面を構成
   ```typescript
   function usePagination() { /* ページネーションロジック */ }
   function useSearch() { /* 検索ロジック */ }
   function useSort() { /* ソートロジック */ }

   // 合成
   function useUserList() {
     const pagination = usePagination();
     const search = useSearch();
     const sort = useSort();
     // 組み合わせて返す
   }
   ```

**目安**: ViewModel の状態が 5 個以上、メソッドが 8 個以上になったら分割を検討する。

### Q4. MVC から MVVM に移行すべきタイミングは？

**A.** 以下の兆候が見られたら移行を検討する:

1. **Controller が肥大化** — 1つの Controller が 300行以上
2. **テストが書けない** — UI をモックしないとテストできないロジックが増えた
3. **リアクティブ要件** — リアルタイム更新、複雑な UI 状態遷移が必要
4. **クロスプラットフォーム** — Web とモバイルでロジックを共有したい

ただし「動いているサーバーサイド MVC」を無理に MVVM に移行する必要はない。フレームワークの規約に従うのが最善。

### Q5. MVP と MVVM のどちらを選ぶべきか？

**A.** 現在（2024年以降）では、ほぼ全てのケースで **MVVM を選ぶべき**。理由:

1. 主要フレームワーク（React, SwiftUI, Compose, Vue, Angular）が全て MVVM 向け
2. リアクティブプログラミングが主流になり、データバインディングが自然
3. MVP の「View インターフェース + Presenter」は MVVM の「ViewModel + バインディング」より冗長

**例外**: フレームワークなしで UI を構築する場合（カスタム描画エンジン等）は MVP が有効。

### Q6. Clean Architecture と MVC/MVVM の関係は？

**A.** 直交する概念であり、組み合わせて使う:

```
┌──────────────────────────────────────────────────┐
│  Clean Architecture の層構造                      │
│                                                  │
│  Presentation Layer ← ここに MVC/MVVM を適用     │
│    ├── View (JSX / Template)                     │
│    └── ViewModel / Controller                    │
│                                                  │
│  Application Layer                               │
│    └── UseCase / Interactor                      │
│                                                  │
│  Domain Layer                                    │
│    ├── Entity                                    │
│    └── Repository Interface                      │
│                                                  │
│  Infrastructure Layer                            │
│    ├── Repository Implementation                 │
│    └── External Services                         │
└──────────────────────────────────────────────────┘
```

MVC/MVVM は **Presentation Layer のパターン**であり、Clean Architecture は **全レイヤーの依存関係ルール**を定義する。両者は補完関係にある。

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| **MVC** | サーバーサイド Web のデファクト。HTTP のリクエスト/レスポンスモデルと自然に対応。Controller は薄く保つ |
| **MVP** | View と Presenter の明確な分離。テスト容易性が高いが、ボイラープレートが多い。レガシー Android で使用 |
| **MVVM** | データバインディングによる宣言的 UI。SPA とモバイルのデファクト。ViewModel は View を知らない |
| **選定基準** | フレームワークの推奨パターンに従うのが最善。迷ったら MVVM |
| **共通原則** | 関心の分離、薄い Controller/ViewModel、テスト可能な設計 |
| **テスト** | MVC: Controller テスト、MVP: Presenter テスト、MVVM: ViewModel テスト。いずれも View なしでテスト可能にする |
| **進化の方向** | MVC(1979) → MVP(1990s) → MVVM(2005) → 宣言的UI(2019+)。View の受動化が一貫したトレンド |
| **アンチパターン** | Fat Controller、God ViewModel、View にビジネスロジック、ViewModel の View 直接操作 |

---

## 次に読むべきガイド

- [01-repository-pattern.md](./01-repository-pattern.md) — データアクセス層の抽象化（MVVM の Model 層設計）
- [02-event-sourcing-cqrs.md](./02-event-sourcing-cqrs.md) — イベント駆動アーキテクチャ（CQRS の Command/Query 分離）
- [../02-behavioral/](../02-behavioral/) — Observer パターン（MVVM のデータバインディングの基盤）
- [../../clean-code-principles/](../../clean-code-principles/) — 関心の分離、SOLID 原則
- [../../system-design-guide/](../../system-design-guide/) — アーキテクチャ設計の全体像

---

## 参考文献

1. **Trygve Reenskaug** — "The original MVC reports" (1979) — https://folk.universitetetioslo.no/trygver/themes/mvc/mvc-index.html
2. **Martin Fowler** — "GUI Architectures" — https://martinfowler.com/eaaDev/uiArchs.html
3. **Microsoft** — "The MVVM Pattern" — https://learn.microsoft.com/en-us/dotnet/architecture/maui/mvvm
4. **Apple Developer Documentation** — "Model-View-ViewModel" — https://developer.apple.com/documentation/swiftui/model-data
5. **Android Developers** — "Guide to app architecture" — https://developer.android.com/topic/architecture
6. **Josh W. Comeau** — "The Wave of React" — Custom Hooks as ViewModel パターンの解説
7. **Robert C. Martin** — "Clean Architecture" (2017) — Presentation Layer パターンと Clean Architecture の関係

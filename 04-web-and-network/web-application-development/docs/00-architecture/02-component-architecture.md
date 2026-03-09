# コンポーネント設計

> コンポーネント設計はUIの再利用性と保守性を決定づける。Atomic Design、Container/Presentational、Compound Components、Headless UIまで、スケーラブルなコンポーネントアーキテクチャの全パターンを習得する。

## この章で学ぶこと

- [ ] コンポーネント分割の原則と粒度設計を理解する
- [ ] 主要なコンポーネント設計パターンを把握する
- [ ] Headless UIとコンポーネントライブラリの活用を学ぶ
- [ ] Props設計の原則とパターンを習得する
- [ ] Server/Clientコンポーネント境界の最適化を学ぶ
- [ ] コンポーネントのテスト戦略を理解する
- [ ] パフォーマンス最適化のためのコンポーネント設計を把握する
- [ ] 大規模アプリケーションにおけるコンポーネント管理手法を学ぶ

## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- プロジェクト構成とFeature-based設計 — [プロジェクト構成](./01-project-structure.md)
- React の基本（JSX、Props、State、Hooks）
- TypeScript の型システム（interface、type、Generics の基礎）

---

## 1. コンポーネント分割の原則

### 1.1 単一責任原則（SRP）

コンポーネント設計における最も重要な原則は、1つのコンポーネントが1つの責任のみを持つことである。これにより、コンポーネントの理解・テスト・保守が容易になる。

```typescript
// ============================================
// アンチパターン: 1つのコンポーネントに全ての責任
// ============================================
function UserPage() {
  const [users, setUsers] = useState<User[]>([]);
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('name');
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editForm, setEditForm] = useState<Partial<User>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    // データ取得ロジック
    fetch(`/api/users?page=${page}&filter=${filter}&sort=${sortBy}&q=${searchQuery}`)
      .then(res => res.json())
      .then(data => setUsers(data.users));
  }, [page, filter, sortBy, searchQuery]);

  const handleEdit = (user: User) => { /* 編集ロジック */ };
  const handleDelete = (id: string) => { /* 削除ロジック */ };
  const handleSearch = (query: string) => { /* 検索ロジック */ };
  const validateForm = () => { /* バリデーション */ };

  // 500行以上のJSX...
  return (
    <div>
      {/* 検索バー、フィルター、テーブル、ページネーション、
          編集モーダル、確認ダイアログ、通知 etc. */}
    </div>
  );
}

// ============================================
// 推奨パターン: 責任ごとにコンポーネントを分割
// ============================================

// ページコンポーネント（構成のみ担当）
function UserPage() {
  return (
    <PageLayout title="ユーザー管理">
      <UserSearchBar />
      <UserFilters />
      <UserTableContainer />
      <Pagination />
    </PageLayout>
  );
}

// 検索バー（検索機能のみ担当）
function UserSearchBar() {
  const [query, setQuery] = useQueryParam('q', '');
  const debouncedQuery = useDebounce(query, 300);

  return (
    <SearchInput
      value={query}
      onChange={setQuery}
      placeholder="ユーザーを検索..."
    />
  );
}

// フィルター（フィルタリングのみ担当）
function UserFilters() {
  const [filter, setFilter] = useQueryParam('filter', 'all');

  return (
    <FilterBar>
      <FilterChip value="all" active={filter === 'all'} onClick={() => setFilter('all')}>
        すべて
      </FilterChip>
      <FilterChip value="active" active={filter === 'active'} onClick={() => setFilter('active')}>
        アクティブ
      </FilterChip>
      <FilterChip value="inactive" active={filter === 'inactive'} onClick={() => setFilter('inactive')}>
        非アクティブ
      </FilterChip>
    </FilterBar>
  );
}

// テーブルコンテナ（データ取得とテーブル表示の橋渡し）
function UserTableContainer() {
  const { data: users, isLoading, error } = useUsers();

  if (error) return <ErrorMessage error={error} />;
  if (isLoading) return <TableSkeleton rows={10} columns={5} />;

  return <UserTable users={users} />;
}
```

### 1.2 分割の判断基準

コンポーネントをいつ分割すべきかの判断基準を明確にしておくことが重要である。

```
分割すべきサイン:
  ✓ コンポーネントが50行を超えている
  ✓ 同じUIパターンが2回以上出現している
  ✓ テストしたい単位が明確に存在する
  ✓ データ取得とUIレンダリングが混在している
  ✓ 複数の状態が独立して管理されている
  ✓ コンポーネント名に「And」が入りそうになる
  ✓ JSXの中に複雑な条件分岐がある

分割しすぎの兆候:
  ✗ props が10個以上のバケツリレーが発生
  ✗ 1つの変更で5ファイル以上の修正が必要
  ✗ コンポーネント名が抽象的すぎる（Wrapper, Handler, Manager）
  ✗ コンポーネントが単なるHTML要素の薄いラッパー
  ✗ 親子間で大量のコールバックを受け渡している
  ✗ ファイル数が多すぎて探すのに時間がかかる
```

### 1.3 コンポーネントの粒度設計

コンポーネントの粒度は、プロジェクトの規模や要件に応じて調整する必要がある。粒度の基準を明確にしておくことで、チーム全体で一貫した設計が可能になる。

```typescript
// ============================================
// 粒度レベル1: プリミティブコンポーネント
// HTML要素を拡張した最小単位
// ============================================
interface TextInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
}

function TextInput({ label, error, helperText, id, ...props }: TextInputProps) {
  const inputId = id ?? useId();
  return (
    <div className="flex flex-col gap-1">
      {label && (
        <label htmlFor={inputId} className="text-sm font-medium text-gray-700">
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={cn(
          'rounded-md border px-3 py-2 text-sm',
          error ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 focus:ring-blue-500'
        )}
        aria-invalid={!!error}
        aria-describedby={error ? `${inputId}-error` : helperText ? `${inputId}-helper` : undefined}
        {...props}
      />
      {error && (
        <p id={`${inputId}-error`} className="text-sm text-red-500" role="alert">
          {error}
        </p>
      )}
      {!error && helperText && (
        <p id={`${inputId}-helper`} className="text-sm text-gray-500">
          {helperText}
        </p>
      )}
    </div>
  );
}

// ============================================
// 粒度レベル2: 複合コンポーネント
// プリミティブを組み合わせた機能単位
// ============================================
interface SearchFormProps {
  onSearch: (query: string, filters: SearchFilters) => void;
  defaultQuery?: string;
  categories: Category[];
}

function SearchForm({ onSearch, defaultQuery = '', categories }: SearchFormProps) {
  const [query, setQuery] = useState(defaultQuery);
  const [filters, setFilters] = useState<SearchFilters>({});

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(query, filters);
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 items-end">
      <TextInput
        label="検索"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="キーワードを入力..."
      />
      <Select
        label="カテゴリ"
        options={categories}
        value={filters.category}
        onChange={(value) => setFilters(prev => ({ ...prev, category: value }))}
      />
      <Button type="submit" variant="primary">
        検索
      </Button>
    </form>
  );
}

// ============================================
// 粒度レベル3: ドメインコンポーネント
// ビジネスロジックを含む特定ドメインの単位
// ============================================
function UserProfile({ userId }: { userId: string }) {
  const { data: user, isLoading } = useUser(userId);
  const { mutate: updateUser } = useUpdateUser();
  const [isEditing, setIsEditing] = useState(false);

  if (isLoading) return <ProfileSkeleton />;
  if (!user) return <NotFound message="ユーザーが見つかりません" />;

  return (
    <Card>
      <Card.Header>
        <Avatar src={user.avatar} alt={user.name} size="lg" />
        <div>
          <h2 className="text-xl font-bold">{user.name}</h2>
          <Badge variant={user.role === 'admin' ? 'primary' : 'secondary'}>
            {user.role}
          </Badge>
        </div>
        <Button variant="ghost" onClick={() => setIsEditing(true)}>
          編集
        </Button>
      </Card.Header>
      <Card.Body>
        <UserProfileDetails user={user} />
      </Card.Body>
      {isEditing && (
        <UserEditModal
          user={user}
          onSave={(data) => {
            updateUser({ id: userId, ...data });
            setIsEditing(false);
          }}
          onClose={() => setIsEditing(false)}
        />
      )}
    </Card>
  );
}

// ============================================
// 粒度レベル4: ページ/レイアウトコンポーネント
// ドメインコンポーネントを組み合わせた画面レベル
// ============================================
function DashboardPage() {
  return (
    <DashboardLayout>
      <DashboardLayout.Sidebar>
        <Navigation />
      </DashboardLayout.Sidebar>
      <DashboardLayout.Main>
        <PageHeader
          title="ダッシュボード"
          actions={<DateRangePicker />}
        />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <KPICard metric="totalUsers" />
          <KPICard metric="activeUsers" />
          <KPICard metric="revenue" />
          <KPICard metric="conversionRate" />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          <RecentActivityFeed />
          <SalesChart period="monthly" />
        </div>
      </DashboardLayout.Main>
    </DashboardLayout>
  );
}
```

### 1.4 Atomic Design

Atomic Designは、UIをAtoms（原子）、Molecules（分子）、Organisms（有機体）、Templates（テンプレート）、Pages（ページ）の5段階で構成するデザインシステムの方法論である。

```
Atomic Design の5階層:

  Atoms（原子）:
  → 最も小さなUI単位
  → Button, Input, Label, Icon, Badge, Avatar
  → それ以上分割できない要素
  → デザイントークン（色、フォント、スペーシング）を直接参照

  Molecules（分子）:
  → Atomsを組み合わせた機能単位
  → SearchBar = Input + Button + Icon
  → FormField = Label + Input + ErrorMessage
  → 1つの明確な機能を持つ

  Organisms（有機体）:
  → Molecules + Atoms で構成される複雑なUI
  → Header = Logo + Navigation + SearchBar + UserMenu
  → ProductCard = Image + Title + Price + AddToCartButton
  → 独立してUIとして成立する

  Templates（テンプレート）:
  → ページのレイアウト構造を定義
  → コンテンツのプレースホルダーを配置
  → データなしの骨組み

  Pages（ページ）:
  → Templatesに実データを流し込んだもの
  → 実際のユーザーが見る最終的な画面
```

```typescript
// ============================================
// Atomic Design の実装例
// ============================================

// --- Atom ---
function Badge({ children, variant = 'default' }: {
  children: ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'error';
}) {
  const variantClasses = {
    default: 'bg-gray-100 text-gray-800',
    success: 'bg-green-100 text-green-800',
    warning: 'bg-yellow-100 text-yellow-800',
    error: 'bg-red-100 text-red-800',
  };

  return (
    <span className={cn('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium', variantClasses[variant])}>
      {children}
    </span>
  );
}

// --- Molecule ---
function UserInfo({ name, email, role }: {
  name: string;
  email: string;
  role: 'admin' | 'user' | 'guest';
}) {
  const roleBadgeVariant = {
    admin: 'error' as const,
    user: 'success' as const,
    guest: 'default' as const,
  };

  return (
    <div className="flex items-center gap-3">
      <Avatar name={name} size="md" />
      <div>
        <p className="font-medium text-gray-900">{name}</p>
        <p className="text-sm text-gray-500">{email}</p>
      </div>
      <Badge variant={roleBadgeVariant[role]}>{role}</Badge>
    </div>
  );
}

// --- Organism ---
function UserListTable({ users, onEdit, onDelete }: {
  users: User[];
  onEdit: (user: User) => void;
  onDelete: (userId: string) => void;
}) {
  return (
    <Table>
      <Table.Header>
        <Table.Row>
          <Table.Head>ユーザー</Table.Head>
          <Table.Head>ステータス</Table.Head>
          <Table.Head>登録日</Table.Head>
          <Table.Head>操作</Table.Head>
        </Table.Row>
      </Table.Header>
      <Table.Body>
        {users.map(user => (
          <Table.Row key={user.id}>
            <Table.Cell>
              <UserInfo name={user.name} email={user.email} role={user.role} />
            </Table.Cell>
            <Table.Cell>
              <Badge variant={user.isActive ? 'success' : 'default'}>
                {user.isActive ? 'アクティブ' : '非アクティブ'}
              </Badge>
            </Table.Cell>
            <Table.Cell>{formatDate(user.createdAt)}</Table.Cell>
            <Table.Cell>
              <div className="flex gap-2">
                <IconButton icon="edit" onClick={() => onEdit(user)} label="編集" />
                <IconButton icon="delete" onClick={() => onDelete(user.id)} label="削除" variant="destructive" />
              </div>
            </Table.Cell>
          </Table.Row>
        ))}
      </Table.Body>
    </Table>
  );
}

// --- Template ---
function AdminListTemplate({ children }: { children: ReactNode }) {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        {/* ヘッダースロット */}
      </div>
      <div className="flex gap-4">
        <div className="flex-1">
          {/* メインコンテンツスロット */}
          {children}
        </div>
        <div className="w-64">
          {/* サイドバースロット */}
        </div>
      </div>
    </div>
  );
}
```

### 1.5 コンポーネント分割のベストプラクティス比較表

| 基準 | 分割する | 分割しない |
|------|---------|-----------|
| 行数 | 50行以上 | 30行以下 |
| 責任 | 複数の関心事 | 単一の関心事 |
| 再利用 | 2箇所以上で使用 | 1箇所でのみ使用 |
| テスト | 独立テストが必要 | 親と一緒にテスト |
| 状態 | 独立した状態管理 | 親の状態に依存 |
| 変更頻度 | 他と異なる変更頻度 | 同じタイミングで変更 |
| チーム | 異なるチームが担当 | 同じチームが担当 |

---

## 2. Container / Presentational パターン

### 2.1 パターンの基本概念

Container/Presentationalパターンは、コンポーネントをロジック担当（Container）と表示担当（Presentational）に分離する設計パターンである。Dan AbramovがReactコミュニティに広めたパターンで、関心の分離を実現する最も基本的な手法の1つである。

```
Container（ロジック担当）:
  → データ取得、状態管理、イベントハンドリング
  → UIを持たない（Presentationalに委譲）
  → カスタムフックとして実装することも多い
  → 副作用（API呼び出し、ストレージアクセス等）を集約

Presentational（表示担当）:
  → propsを受け取って表示するだけ
  → 内部状態は最小限（UIの開閉、ホバー状態等）
  → テストが容易（propsを渡すだけ）
  → Storybookでのドキュメント化が容易
  → 再利用性が高い
```

### 2.2 実装パターン

```typescript
// ============================================
// パターン1: クラシックなContainer/Presentational
// ============================================

// --- Container ---
function UserListContainer() {
  const { data: users, isLoading, error } = useUsers();
  const [filter, setFilter] = useState<UserFilter>('all');
  const [sortBy, setSortBy] = useState<SortKey>('name');
  const { mutate: deleteUser } = useDeleteUser();

  const filteredUsers = useMemo(() => {
    if (!users) return [];
    return users
      .filter(u => filter === 'all' ? true : u.role === filter)
      .sort((a, b) => a[sortBy].localeCompare(b[sortBy]));
  }, [users, filter, sortBy]);

  const handleDelete = useCallback(async (userId: string) => {
    if (window.confirm('本当に削除しますか？')) {
      await deleteUser(userId);
    }
  }, [deleteUser]);

  return (
    <UserListView
      users={filteredUsers}
      isLoading={isLoading}
      error={error}
      filter={filter}
      sortBy={sortBy}
      onFilterChange={setFilter}
      onSortChange={setSortBy}
      onDelete={handleDelete}
    />
  );
}

// --- Presentational ---
interface UserListViewProps {
  users: User[];
  isLoading: boolean;
  error: Error | null;
  filter: UserFilter;
  sortBy: SortKey;
  onFilterChange: (filter: UserFilter) => void;
  onSortChange: (sort: SortKey) => void;
  onDelete: (userId: string) => void;
}

function UserListView({
  users,
  isLoading,
  error,
  filter,
  sortBy,
  onFilterChange,
  onSortChange,
  onDelete,
}: UserListViewProps) {
  if (error) {
    return (
      <Alert variant="error">
        <AlertTitle>エラー</AlertTitle>
        <AlertDescription>{error.message}</AlertDescription>
      </Alert>
    );
  }

  if (isLoading) {
    return <TableSkeleton rows={10} columns={4} />;
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <FilterBar value={filter} onChange={onFilterChange} />
        <SortSelect value={sortBy} onChange={onSortChange} />
      </div>
      {users.length === 0 ? (
        <EmptyState
          icon="users"
          title="ユーザーが見つかりません"
          description="検索条件を変更してください"
        />
      ) : (
        <ul className="divide-y divide-gray-200">
          {users.map(user => (
            <li key={user.id} className="py-4">
              <UserCard user={user} onDelete={() => onDelete(user.id)} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ============================================
// パターン2: カスタムフックによる分離（現代的アプローチ）
// ============================================

// カスタムフック = Container の役割
function useUserList() {
  const { data: users, isLoading, error } = useUsers();
  const [filter, setFilter] = useState<UserFilter>('all');
  const [sortBy, setSortBy] = useState<SortKey>('name');
  const { mutate: deleteUser } = useDeleteUser();

  const filteredUsers = useMemo(() => {
    if (!users) return [];
    return users
      .filter(u => filter === 'all' ? true : u.role === filter)
      .sort((a, b) => a[sortBy].localeCompare(b[sortBy]));
  }, [users, filter, sortBy]);

  const handleDelete = useCallback(async (userId: string) => {
    if (window.confirm('本当に削除しますか？')) {
      await deleteUser(userId);
    }
  }, [deleteUser]);

  return {
    users: filteredUsers,
    isLoading,
    error,
    filter,
    sortBy,
    setFilter,
    setSortBy,
    handleDelete,
  };
}

// コンポーネント側は表示に集中
function UserList() {
  const {
    users,
    isLoading,
    error,
    filter,
    sortBy,
    setFilter,
    setSortBy,
    handleDelete,
  } = useUserList();

  // 表示ロジックのみ
  if (error) return <ErrorMessage error={error} />;
  if (isLoading) return <LoadingSpinner />;

  return (
    <div>
      <FilterBar value={filter} onChange={setFilter} />
      <SortSelect value={sortBy} onChange={setSortBy} />
      <UserListView users={users} onDelete={handleDelete} />
    </div>
  );
}

// ============================================
// パターン3: React Server Components による自然な分離
// ============================================

// Server Component = Container（データ取得）
// app/users/page.tsx
async function UsersPage() {
  const users = await prisma.user.findMany({
    orderBy: { createdAt: 'desc' },
    include: { profile: true },
  });

  return (
    <PageLayout>
      <PageHeader title="ユーザー一覧" />
      {/* Client Componentに表示を委譲 */}
      <UserListClient initialUsers={users} />
    </PageLayout>
  );
}

// Client Component = Presentational（インタラクション）
'use client';
function UserListClient({ initialUsers }: { initialUsers: User[] }) {
  const [filter, setFilter] = useState<UserFilter>('all');
  const filtered = initialUsers.filter(u =>
    filter === 'all' ? true : u.role === filter
  );

  return (
    <>
      <FilterBar value={filter} onChange={setFilter} />
      <UserGrid users={filtered} />
    </>
  );
}
```

### 2.3 Container/Presentational パターンの使い分け

| アプローチ | メリット | デメリット | 適用場面 |
|-----------|---------|-----------|---------|
| クラシックContainer | 明確な分離、テスト容易 | ファイル数増加 | 大規模チーム |
| カスタムフック | 柔軟、再利用容易 | フックの依存管理 | 中規模プロジェクト |
| RSC分離 | 自然な分離、パフォーマンス | Next.js依存 | Next.js App Router |

---

## 3. Compound Components パターン

### 3.1 パターンの概要

Compound Componentsは、関連するコンポーネント群を1つのまとまりとして提供するパターンである。親コンポーネントが状態を管理し、子コンポーネントがその状態を暗黙的に共有する。HTMLの `<select>` と `<option>` の関係に類似している。

```typescript
// ============================================
// 使い方のイメージ: 宣言的で直感的なAPI
// ============================================

// Tabsコンポーネントの使用例
<Tabs defaultValue="profile">
  <Tabs.List>
    <Tabs.Trigger value="profile">プロフィール</Tabs.Trigger>
    <Tabs.Trigger value="settings">設定</Tabs.Trigger>
    <Tabs.Trigger value="billing">請求</Tabs.Trigger>
  </Tabs.List>
  <Tabs.Content value="profile">
    <ProfileForm />
  </Tabs.Content>
  <Tabs.Content value="settings">
    <SettingsForm />
  </Tabs.Content>
  <Tabs.Content value="billing">
    <BillingInfo />
  </Tabs.Content>
</Tabs>

// Accordionコンポーネントの使用例
<Accordion type="single" defaultValue="item-1">
  <Accordion.Item value="item-1">
    <Accordion.Trigger>セクション1</Accordion.Trigger>
    <Accordion.Content>セクション1の内容</Accordion.Content>
  </Accordion.Item>
  <Accordion.Item value="item-2">
    <Accordion.Trigger>セクション2</Accordion.Trigger>
    <Accordion.Content>セクション2の内容</Accordion.Content>
  </Accordion.Item>
</Accordion>

// Dropdownメニューの使用例
<DropdownMenu>
  <DropdownMenu.Trigger>
    <Button variant="ghost">メニュー</Button>
  </DropdownMenu.Trigger>
  <DropdownMenu.Content>
    <DropdownMenu.Item onSelect={() => navigate('/profile')}>
      プロフィール
    </DropdownMenu.Item>
    <DropdownMenu.Separator />
    <DropdownMenu.Item onSelect={handleLogout} variant="destructive">
      ログアウト
    </DropdownMenu.Item>
  </DropdownMenu.Content>
</DropdownMenu>
```

### 3.2 Tabsコンポーネントの実装

```typescript
// ============================================
// Compound Components の完全な実装例: Tabs
// ============================================

// --- 型定義 ---
interface TabsContextType {
  activeTab: string;
  setActiveTab: (value: string) => void;
  orientation: 'horizontal' | 'vertical';
}

// --- Context ---
const TabsContext = createContext<TabsContextType | null>(null);

function useTabsContext() {
  const context = useContext(TabsContext);
  if (!context) {
    throw new Error('Tabs のサブコンポーネントは <Tabs> 内で使用してください');
  }
  return context;
}

// --- 親コンポーネント ---
interface TabsProps {
  defaultValue: string;
  value?: string;
  onValueChange?: (value: string) => void;
  orientation?: 'horizontal' | 'vertical';
  children: ReactNode;
}

function Tabs({
  defaultValue,
  value: controlledValue,
  onValueChange,
  orientation = 'horizontal',
  children,
}: TabsProps) {
  const [internalValue, setInternalValue] = useState(defaultValue);
  const isControlled = controlledValue !== undefined;
  const activeTab = isControlled ? controlledValue : internalValue;

  const setActiveTab = useCallback((newValue: string) => {
    if (!isControlled) {
      setInternalValue(newValue);
    }
    onValueChange?.(newValue);
  }, [isControlled, onValueChange]);

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab, orientation }}>
      <div
        className={cn(
          'flex',
          orientation === 'horizontal' ? 'flex-col' : 'flex-row'
        )}
      >
        {children}
      </div>
    </TabsContext.Provider>
  );
}

// --- Tabs.List ---
Tabs.List = function TabsList({ children, className }: {
  children: ReactNode;
  className?: string;
}) {
  const { orientation } = useTabsContext();

  return (
    <div
      role="tablist"
      aria-orientation={orientation}
      className={cn(
        'flex gap-1',
        orientation === 'horizontal'
          ? 'border-b border-gray-200'
          : 'flex-col border-r border-gray-200',
        className
      )}
    >
      {children}
    </div>
  );
};

// --- Tabs.Trigger ---
Tabs.Trigger = function TabsTrigger({ value, children, disabled = false, className }: {
  value: string;
  children: ReactNode;
  disabled?: boolean;
  className?: string;
}) {
  const { activeTab, setActiveTab } = useTabsContext();
  const isActive = activeTab === value;
  const ref = useRef<HTMLButtonElement>(null);

  // キーボードナビゲーション
  const handleKeyDown = (e: React.KeyboardEvent) => {
    const triggers = ref.current?.parentElement?.querySelectorAll('[role="tab"]');
    if (!triggers) return;

    const currentIndex = Array.from(triggers).indexOf(ref.current!);
    let nextIndex: number;

    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        e.preventDefault();
        nextIndex = (currentIndex + 1) % triggers.length;
        (triggers[nextIndex] as HTMLElement).focus();
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        e.preventDefault();
        nextIndex = (currentIndex - 1 + triggers.length) % triggers.length;
        (triggers[nextIndex] as HTMLElement).focus();
        break;
      case 'Home':
        e.preventDefault();
        (triggers[0] as HTMLElement).focus();
        break;
      case 'End':
        e.preventDefault();
        (triggers[triggers.length - 1] as HTMLElement).focus();
        break;
    }
  };

  return (
    <button
      ref={ref}
      role="tab"
      aria-selected={isActive}
      aria-controls={`tabpanel-${value}`}
      id={`tab-${value}`}
      tabIndex={isActive ? 0 : -1}
      disabled={disabled}
      onClick={() => !disabled && setActiveTab(value)}
      onKeyDown={handleKeyDown}
      className={cn(
        'px-4 py-2 text-sm font-medium transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
        isActive
          ? 'border-b-2 border-blue-500 text-blue-600'
          : 'text-gray-500 hover:text-gray-700',
        disabled && 'opacity-50 cursor-not-allowed',
        className
      )}
    >
      {children}
    </button>
  );
};

// --- Tabs.Content ---
Tabs.Content = function TabsContent({ value, children, className }: {
  value: string;
  children: ReactNode;
  className?: string;
}) {
  const { activeTab } = useTabsContext();
  const isActive = activeTab === value;

  if (!isActive) return null;

  return (
    <div
      role="tabpanel"
      id={`tabpanel-${value}`}
      aria-labelledby={`tab-${value}`}
      tabIndex={0}
      className={cn('p-4 focus-visible:outline-none', className)}
    >
      {children}
    </div>
  );
};
```

### 3.3 Accordionコンポーネントの実装

```typescript
// ============================================
// Compound Components: Accordion
// ============================================

type AccordionType = 'single' | 'multiple';

interface AccordionContextType {
  type: AccordionType;
  openItems: Set<string>;
  toggleItem: (value: string) => void;
}

const AccordionContext = createContext<AccordionContextType | null>(null);

function useAccordionContext() {
  const ctx = useContext(AccordionContext);
  if (!ctx) throw new Error('Accordion サブコンポーネントは <Accordion> 内で使用してください');
  return ctx;
}

// --- Accordion 本体 ---
interface AccordionProps {
  type?: AccordionType;
  defaultValue?: string | string[];
  children: ReactNode;
}

function Accordion({ type = 'single', defaultValue, children }: AccordionProps) {
  const [openItems, setOpenItems] = useState<Set<string>>(() => {
    if (!defaultValue) return new Set();
    return new Set(Array.isArray(defaultValue) ? defaultValue : [defaultValue]);
  });

  const toggleItem = useCallback((value: string) => {
    setOpenItems(prev => {
      const next = new Set(prev);
      if (next.has(value)) {
        next.delete(value);
      } else {
        if (type === 'single') {
          next.clear();
        }
        next.add(value);
      }
      return next;
    });
  }, [type]);

  return (
    <AccordionContext.Provider value={{ type, openItems, toggleItem }}>
      <div className="divide-y divide-gray-200 border rounded-lg">
        {children}
      </div>
    </AccordionContext.Provider>
  );
}

// --- AccordionItem ---
const AccordionItemContext = createContext<string>('');

Accordion.Item = function AccordionItem({ value, children }: {
  value: string;
  children: ReactNode;
}) {
  return (
    <AccordionItemContext.Provider value={value}>
      <div className="border-b last:border-b-0">{children}</div>
    </AccordionItemContext.Provider>
  );
};

// --- AccordionTrigger ---
Accordion.Trigger = function AccordionTrigger({ children }: {
  children: ReactNode;
}) {
  const { openItems, toggleItem } = useAccordionContext();
  const value = useContext(AccordionItemContext);
  const isOpen = openItems.has(value);

  return (
    <button
      type="button"
      aria-expanded={isOpen}
      aria-controls={`accordion-content-${value}`}
      onClick={() => toggleItem(value)}
      className="flex w-full items-center justify-between py-4 px-6 text-left font-medium transition-colors hover:bg-gray-50"
    >
      {children}
      <ChevronIcon
        className={cn(
          'h-4 w-4 transition-transform duration-200',
          isOpen && 'rotate-180'
        )}
      />
    </button>
  );
};

// --- AccordionContent ---
Accordion.Content = function AccordionContent({ children }: {
  children: ReactNode;
}) {
  const { openItems } = useAccordionContext();
  const value = useContext(AccordionItemContext);
  const isOpen = openItems.has(value);
  const contentRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number>(0);

  useEffect(() => {
    if (contentRef.current) {
      setHeight(contentRef.current.scrollHeight);
    }
  }, [children]);

  return (
    <div
      id={`accordion-content-${value}`}
      role="region"
      aria-labelledby={`accordion-trigger-${value}`}
      className="overflow-hidden transition-all duration-200"
      style={{ height: isOpen ? height : 0 }}
    >
      <div ref={contentRef} className="px-6 pb-4">
        {children}
      </div>
    </div>
  );
};
```

### 3.4 Compound Components パターンの利点と注意点

```
利点:
  ✓ 宣言的で直感的なAPI
  ✓ 柔軟なレイアウトカスタマイズ
  ✓ 関連コンポーネント間の暗黙的な状態共有
  ✓ コンポーネント間のpropsバケツリレーを回避
  ✓ 各サブコンポーネントの独立したスタイリング

注意点:
  ✗ Context の過度なネストによるパフォーマンス問題
  ✗ TypeScript の型定義が複雑になりがち
  ✗ 関数コンポーネントの静的プロパティ（displayName等）の管理
  ✗ 子コンポーネントの使い方を制約しにくい

実装パターンの選択:
  静的プロパティ方式:
    Tabs.List, Tabs.Trigger, Tabs.Content
    → シンプルで直感的
    → tree-shaking が効かない場合がある

  名前付きエクスポート方式:
    TabsList, TabsTrigger, TabsContent
    → tree-shaking に対応
    → import が冗長になる

  推奨: shadcn/ui スタイルの名前付きエクスポート
    import { Tabs, TabsList, TabsTrigger, TabsContent } from './tabs';
```

---

## 4. Headless UI

### 4.1 Headless UIの概念

Headless UIとは、ロジック・状態管理・アクセシビリティのみを提供し、スタイルは一切含まないUIコンポーネントのアーキテクチャである。これにより、見た目の完全なカスタマイズを保ちながら、複雑なインタラクションロジックとアクセシビリティを再利用できる。

```
Headless UI の思想:
  → ロジック層とプレゼンテーション層の完全な分離
  → アクセシビリティ（WAI-ARIA）の標準準拠
  → キーボードナビゲーションの完全サポート
  → フォーカス管理の自動化
  → スタイルの一切を消費者に委ねる

従来のUIライブラリの課題:
  → 見た目のカスタマイズが困難
  → CSSの上書きが複雑（!important 地獄）
  → デザインシステムとの統合が難しい
  → バンドルサイズが大きい

Headless UIが解決する問題:
  → スタイルの制約がゼロ
  → 既存のCSSフレームワークと自然に統合
  → 必要なコンポーネントのみ利用可能
  → アクセシビリティの自力実装が不要
```

### 4.2 主要なHeadless UIライブラリの比較

```
ライブラリ比較:

  ┌─────────────┬──────────┬─────────────┬──────────┬─────────┐
  │ ライブラリ   │ 開発元   │ 特徴         │ サイズ    │ 推奨度  │
  ├─────────────┼──────────┼─────────────┼──────────┼─────────┤
  │ Radix UI    │ WorkOS   │ 最も人気、    │ 中       │ ★★★★★ │
  │             │          │ shadcn/ui   │          │         │
  │             │          │ のベース     │          │         │
  ├─────────────┼──────────┼─────────────┼──────────┼─────────┤
  │ Headless UI │ Tailwind │ Tailwind    │ 小       │ ★★★★☆ │
  │             │ Labs     │ との親和性高 │          │         │
  ├─────────────┼──────────┼─────────────┼──────────┼─────────┤
  │ React Aria  │ Adobe    │ アクセシビリ │ 大       │ ★★★★★ │
  │             │          │ ティ最高    │          │         │
  ├─────────────┼──────────┼─────────────┼──────────┼─────────┤
  │ Ariakit     │ OSS      │ 軽量、      │ 小       │ ★★★★☆ │
  │             │          │ コンポーザ   │          │         │
  │             │          │ ブル        │          │         │
  ├─────────────┼──────────┼─────────────┼──────────┼─────────┤
  │ Ark UI      │ Chakra   │ Zag.js      │ 中       │ ★★★☆☆ │
  │             │          │ ベース、     │          │         │
  │             │          │ FW非依存    │          │         │
  └─────────────┴──────────┴─────────────┴──────────┴─────────┘
```

### 4.3 Radix UIの実践的な使い方

```typescript
// ============================================
// Radix UI + Tailwind CSS でカスタムDialogを構築
// ============================================

import * as Dialog from '@radix-ui/react-dialog';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';

// shadcn/ui スタイルのDialog実装
const DialogRoot = Dialog.Root;
const DialogTrigger = Dialog.Trigger;

const DialogPortal = Dialog.Portal;

const DialogOverlay = forwardRef<
  React.ElementRef<typeof Dialog.Overlay>,
  React.ComponentPropsWithoutRef<typeof Dialog.Overlay>
>(({ className, ...props }, ref) => (
  <Dialog.Overlay
    ref={ref}
    className={cn(
      'fixed inset-0 z-50 bg-black/50',
      'data-[state=open]:animate-in data-[state=open]:fade-in-0',
      'data-[state=closed]:animate-out data-[state=closed]:fade-out-0',
      className
    )}
    {...props}
  />
));
DialogOverlay.displayName = Dialog.Overlay.displayName;

const DialogContent = forwardRef<
  React.ElementRef<typeof Dialog.Content>,
  React.ComponentPropsWithoutRef<typeof Dialog.Content>
>(({ className, children, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay />
    <Dialog.Content
      ref={ref}
      className={cn(
        'fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2',
        'w-full max-w-lg rounded-lg bg-white p-6 shadow-xl',
        'data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
        'data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95',
        className
      )}
      {...props}
    >
      {children}
      <Dialog.Close className="absolute right-4 top-4 rounded-sm opacity-70 hover:opacity-100 focus:ring-2">
        <X className="h-4 w-4" />
        <span className="sr-only">閉じる</span>
      </Dialog.Close>
    </Dialog.Content>
  </DialogPortal>
));
DialogContent.displayName = Dialog.Content.displayName;

const DialogHeader = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn('flex flex-col space-y-1.5 text-center sm:text-left', className)} {...props} />
);

const DialogTitle = forwardRef<
  React.ElementRef<typeof Dialog.Title>,
  React.ComponentPropsWithoutRef<typeof Dialog.Title>
>(({ className, ...props }, ref) => (
  <Dialog.Title
    ref={ref}
    className={cn('text-lg font-semibold leading-none tracking-tight', className)}
    {...props}
  />
));

const DialogDescription = forwardRef<
  React.ElementRef<typeof Dialog.Description>,
  React.ComponentPropsWithoutRef<typeof Dialog.Description>
>(({ className, ...props }, ref) => (
  <Dialog.Description
    ref={ref}
    className={cn('text-sm text-gray-500', className)}
    {...props}
  />
));

// --- 使用例 ---
function ConfirmDeleteDialog({ userName, onConfirm }: {
  userName: string;
  onConfirm: () => void;
}) {
  return (
    <DialogRoot>
      <DialogTrigger asChild>
        <Button variant="destructive" size="sm">削除</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>ユーザーの削除</DialogTitle>
          <DialogDescription>
            {userName} を削除しますか？この操作は取り消せません。
          </DialogDescription>
        </DialogHeader>
        <div className="flex justify-end gap-2 mt-6">
          <Dialog.Close asChild>
            <Button variant="outline">キャンセル</Button>
          </Dialog.Close>
          <Button variant="destructive" onClick={onConfirm}>
            削除する
          </Button>
        </div>
      </DialogContent>
    </DialogRoot>
  );
}
```

### 4.4 shadcn/ui の仕組みと活用

```
shadcn/ui の設計思想:
  → Radix UI（Headless）+ Tailwind CSS（スタイル）+ cva（バリアント管理）
  → npm パッケージとしてインストールしない
  → コピー＆ペーストでコンポーネントを追加
  → 完全にカスタマイズ可能
  → node_modules に依存しない
  → コンポーネントのコードが手元にある安心感

セットアップ手順:
  npx shadcn@latest init
  → tailwind.config.js の設定
  → CSS変数の設定（テーマカラー）
  → パス設定（components, lib, utils）

コンポーネント追加:
  npx shadcn@latest add button
  → src/components/ui/button.tsx が生成
  → 中身を自由に編集可能

  npx shadcn@latest add dialog
  → src/components/ui/dialog.tsx が生成
  → Radix UI Dialog をラップしたコンポーネント

  npx shadcn@latest add form
  → src/components/ui/form.tsx が生成
  → React Hook Form + Zod との統合
```

```typescript
// shadcn/ui のButtonコンポーネント（生成されるコード）
import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
```

### 4.5 コンポーネントライブラリの選定ガイド

| カテゴリ | ライブラリ | 特徴 | 推奨シーン |
|---------|-----------|------|-----------|
| フルスタイル | MUI (Material UI) | Material Design準拠、機能豊富 | エンタープライズ、Material好み |
| フルスタイル | Ant Design | エンタープライズ向け、中国発 | 管理画面、ダッシュボード |
| フルスタイル | Chakra UI | DX重視、学習コスト低 | 中小規模、プロトタイプ |
| フルスタイル | Mantine | React特化、モダン設計 | React専用プロジェクト |
| Headless+スタイル | shadcn/ui | Radix + Tailwind | 新規プロジェクト（推奨） |
| Headless+スタイル | Ark UI | Zag.jsベース、FW非依存 | マルチフレームワーク |
| Headlessのみ | Radix UI | 最も人気、高品質 | カスタムデザイン |
| Headlessのみ | React Aria | Adobe製、a11y最高 | アクセシビリティ重視 |
| Headlessのみ | Headless UI | Tailwind Labs製 | Tailwind環境 |

---

## 5. Props設計

### 5.1 Props設計の基本原則

Props設計は、コンポーネントの使いやすさと保守性に直結する。良いProps設計は、APIの一貫性、型安全性、拡張性を実現する。

```typescript
// ============================================
// 原則1: HTML標準属性を拡張する
// ============================================

// 悪い例: 独自のprops名を使い、HTML標準を無視
interface BadButtonProps {
  label: string;
  onPress: () => void;
  isDisabled: boolean;
  buttonType: 'submit' | 'button';
}

// 良い例: HTML標準属性を拡張
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'destructive' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
}

function Button({
  variant = 'default',
  size = 'md',
  isLoading = false,
  leftIcon,
  rightIcon,
  children,
  disabled,
  className,
  ...props // onClick, type, form 等はそのまま透過
}: ButtonProps) {
  return (
    <button
      className={cn(buttonVariants({ variant, size }), className)}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? <Spinner size="sm" /> : leftIcon}
      {children}
      {rightIcon}
    </button>
  );
}

// ============================================
// 原則2: childrenを活用したコンポジション
// ============================================

// 悪い例: props で全てを制御
<Card
  title="ユーザー情報"
  subtitle="基本情報"
  body={<UserDetails user={user} />}
  footer={<Button onClick={onSave}>保存</Button>}
  headerAction={<IconButton icon="edit" />}
/>

// 良い例: children + Compound Components
<Card>
  <Card.Header>
    <Card.Title>ユーザー情報</Card.Title>
    <Card.Description>基本情報</Card.Description>
    <Card.Action>
      <IconButton icon="edit" />
    </Card.Action>
  </Card.Header>
  <Card.Body>
    <UserDetails user={user} />
  </Card.Body>
  <Card.Footer>
    <Button onClick={onSave}>保存</Button>
  </Card.Footer>
</Card>

// ============================================
// 原則3: 条件付きProps（Discriminated Union）
// ============================================

// ステータスに応じて異なるpropsを要求
type NotificationProps =
  | { type: 'success'; message: string }
  | { type: 'error'; message: string; retryAction: () => void }
  | { type: 'warning'; message: string; dismissable?: boolean }
  | { type: 'info'; message: string; link?: { label: string; href: string } };

function Notification(props: NotificationProps) {
  const baseClasses = 'p-4 rounded-lg flex items-start gap-3';

  switch (props.type) {
    case 'success':
      return (
        <div className={cn(baseClasses, 'bg-green-50 text-green-800')}>
          <CheckIcon className="h-5 w-5 text-green-500" />
          <p>{props.message}</p>
        </div>
      );
    case 'error':
      return (
        <div className={cn(baseClasses, 'bg-red-50 text-red-800')}>
          <XCircleIcon className="h-5 w-5 text-red-500" />
          <p>{props.message}</p>
          <Button size="sm" variant="outline" onClick={props.retryAction}>
            再試行
          </Button>
        </div>
      );
    case 'warning':
      return (
        <div className={cn(baseClasses, 'bg-yellow-50 text-yellow-800')}>
          <AlertIcon className="h-5 w-5 text-yellow-500" />
          <p>{props.message}</p>
          {props.dismissable && <CloseButton />}
        </div>
      );
    case 'info':
      return (
        <div className={cn(baseClasses, 'bg-blue-50 text-blue-800')}>
          <InfoIcon className="h-5 w-5 text-blue-500" />
          <p>{props.message}</p>
          {props.link && (
            <a href={props.link.href} className="underline">{props.link.label}</a>
          )}
        </div>
      );
  }
}

// 使い方: 型安全にpropsが制約される
<Notification type="error" message="保存に失敗" retryAction={() => save()} />
// type="error" の場合、retryAction が必須
// type="success" の場合、retryAction は不要
```

### 5.2 Render Props と Slots パターン

```typescript
// ============================================
// Render Props パターン
// ============================================

// DataTableの柔軟なカスタマイズ
interface DataTableProps<T> {
  data: T[];
  columns: ColumnDef<T>[];
  renderRow?: (item: T, index: number) => ReactNode;
  renderEmpty?: () => ReactNode;
  renderHeader?: (column: ColumnDef<T>) => ReactNode;
  renderFooter?: (data: T[]) => ReactNode;
  renderLoading?: () => ReactNode;
  isLoading?: boolean;
}

function DataTable<T extends { id: string }>({
  data,
  columns,
  renderRow,
  renderEmpty,
  renderHeader,
  renderFooter,
  renderLoading,
  isLoading,
}: DataTableProps<T>) {
  if (isLoading && renderLoading) {
    return renderLoading();
  }

  if (data.length === 0 && renderEmpty) {
    return renderEmpty();
  }

  return (
    <table className="w-full border-collapse">
      <thead>
        <tr>
          {columns.map(col => (
            <th key={col.id}>
              {renderHeader ? renderHeader(col) : col.header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((item, index) =>
          renderRow ? (
            <Fragment key={item.id}>{renderRow(item, index)}</Fragment>
          ) : (
            <tr key={item.id}>
              {columns.map(col => (
                <td key={col.id}>{col.cell(item)}</td>
              ))}
            </tr>
          )
        )}
      </tbody>
      {renderFooter && (
        <tfoot>
          <tr><td colSpan={columns.length}>{renderFooter(data)}</td></tr>
        </tfoot>
      )}
    </table>
  );
}

// 使用例
<DataTable
  data={users}
  columns={userColumns}
  renderRow={(user) => (
    <tr className={user.isActive ? 'bg-white' : 'bg-gray-50'}>
      <td><UserInfo user={user} /></td>
      <td>{formatDate(user.createdAt)}</td>
      <td><UserActions user={user} /></td>
    </tr>
  )}
  renderEmpty={() => (
    <EmptyState
      icon="users"
      title="ユーザーが見つかりません"
      action={<Button onClick={onCreateUser}>ユーザーを追加</Button>}
    />
  )}
  renderLoading={() => <TableSkeleton rows={5} columns={3} />}
  isLoading={isLoading}
/>
```

### 5.3 バリアント管理（CVA: Class Variance Authority）

```typescript
// ============================================
// CVA によるバリアント管理の実践
// ============================================
import { cva, type VariantProps } from 'class-variance-authority';

// --- Alert コンポーネント ---
const alertVariants = cva(
  // ベーススタイル（常に適用）
  'relative w-full rounded-lg border p-4 flex items-start gap-3',
  {
    variants: {
      variant: {
        default: 'bg-white border-gray-200 text-gray-900',
        info: 'bg-blue-50 border-blue-200 text-blue-900',
        success: 'bg-green-50 border-green-200 text-green-900',
        warning: 'bg-yellow-50 border-yellow-200 text-yellow-900',
        error: 'bg-red-50 border-red-200 text-red-900',
      },
      size: {
        sm: 'p-3 text-sm',
        md: 'p-4 text-base',
        lg: 'p-6 text-lg',
      },
      dismissable: {
        true: 'pr-10',
        false: '',
      },
    },
    compoundVariants: [
      // 特定の組み合わせに対するスタイル
      {
        variant: 'error',
        size: 'lg',
        className: 'border-2',
      },
    ],
    defaultVariants: {
      variant: 'default',
      size: 'md',
      dismissable: false,
    },
  }
);

interface AlertProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof alertVariants> {
  icon?: ReactNode;
  onDismiss?: () => void;
}

function Alert({
  variant,
  size,
  dismissable,
  icon,
  onDismiss,
  className,
  children,
  ...props
}: AlertProps) {
  return (
    <div
      role="alert"
      className={cn(alertVariants({ variant, size, dismissable }), className)}
      {...props}
    >
      {icon && <span className="flex-shrink-0 mt-0.5">{icon}</span>}
      <div className="flex-1">{children}</div>
      {dismissable && (
        <button
          onClick={onDismiss}
          className="absolute right-2 top-2 rounded-sm opacity-70 hover:opacity-100"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}

// 使用例
<Alert variant="error" size="lg" dismissable onDismiss={() => setVisible(false)}>
  <AlertTitle>エラー</AlertTitle>
  <AlertDescription>データの保存に失敗しました。再試行してください。</AlertDescription>
</Alert>
```

---

## 6. Server / Client コンポーネント境界

### 6.1 Next.js App Router でのコンポーネント設計

Next.js App Routerでは、Server ComponentとClient Componentという2種類のコンポーネントが存在する。この境界を適切に設計することが、パフォーマンスと開発体験の両方に大きく影響する。

```
基本ルール:
  → デフォルトは Server Component
  → 'use client' は必要最小限に
  → Client の境界をなるべく葉（リーフ）に近づける
  → Server Component から Client Component にはシリアライズ可能なpropsのみ渡せる
```

```typescript
// ============================================
// 良い例: Client境界が小さい
// ============================================

// page.tsx (Server Component)
async function ProductPage({ params }: { params: { id: string } }) {
  const product = await getProduct(params.id);
  const reviews = await getReviews(params.id);

  return (
    <div className="max-w-4xl mx-auto py-8">
      {/* Server Component: 静的な部分 */}
      <h1 className="text-3xl font-bold">{product.name}</h1>
      <p className="mt-2 text-gray-600">{product.description}</p>

      {/* Server Component: 画像ギャラリー（静的） */}
      <ProductImageGallery images={product.images} />

      {/* Client Component: インタラクティブな部分のみ */}
      <ProductPrice price={product.price} discount={product.discount} />
      <AddToCartButton productId={product.id} />

      {/* Server Component: レビュー一覧（静的） */}
      <ReviewList reviews={reviews} />

      {/* Client Component: レビュー投稿フォーム */}
      <ReviewForm productId={product.id} />
    </div>
  );
}

// ============================================
// 悪い例: ページ全体がClient
// ============================================

// 'use client';  ← ページ全体をClientにしてしまっている
// function ProductPage({ params }) {
//   const { data: product } = useQuery(...);
//   // 全てがクライアントサイドで実行される
//   // → バンドルサイズ増大、初期表示遅延
// }
```

### 6.2 Server/Client コンポーネントの判断基準

```
Server Component を使う場面:
  → データベースへの直接アクセス
  → サーバーサイドのAPIキーやシークレットの使用
  → 大きな依存パッケージ（マークダウンパーサー、syntax highlighter等）
  → 機密情報の処理
  → SEOが重要なコンテンツ
  → 初期表示パフォーマンスが重要な部分

Client Component を使う場面:
  → useState, useEffect, useReducer などのReact Hooksが必要
  → onClick, onChange 等のイベントハンドラが必要
  → ブラウザAPI（localStorage, navigator, window等）へのアクセス
  → サードパーティのクライアントサイドライブラリ
  → Context Providerの利用
  → アニメーションやトランジション
```

### 6.3 境界設計のパターン

```typescript
// ============================================
// パターン1: インタラクティブな部分だけをClient化
// ============================================

// SearchableList.tsx (Server Component)
async function SearchableList() {
  // サーバーサイドで全データを取得
  const items = await fetchAllItems();

  return (
    <div>
      <h2>アイテム一覧</h2>
      {/* 検索機能だけをClient化 */}
      <SearchFilter items={items} />
    </div>
  );
}

// SearchFilter.tsx (Client Component)
'use client';
function SearchFilter({ items }: { items: Item[] }) {
  const [query, setQuery] = useState('');
  const filtered = items.filter(item =>
    item.name.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <>
      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="検索..."
      />
      <ul>
        {filtered.map(item => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </>
  );
}

// ============================================
// パターン2: Provider のClient境界
// ============================================

// providers.tsx (Client Component)
'use client';
import { ThemeProvider } from 'next-themes';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';

const queryClient = new QueryClient();

export function Providers({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider attribute="class" defaultTheme="system">
        {children}
      </ThemeProvider>
    </QueryClientProvider>
  );
}

// layout.tsx (Server Component)
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ja">
      <body>
        <Providers>
          {/* childrenはServer Componentのまま */}
          {children}
        </Providers>
      </body>
    </html>
  );
}

// ============================================
// パターン3: Server Component を Children として渡す
// ============================================

// ClientWrapper.tsx (Client Component)
'use client';
function Sidebar({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <aside className={cn('transition-all', isOpen ? 'w-64' : 'w-16')}>
      <button onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? '閉じる' : '開く'}
      </button>
      {isOpen && children}
    </aside>
  );
}

// page.tsx (Server Component)
async function DashboardPage() {
  const navItems = await getNavItems(); // サーバーサイドで取得

  return (
    <div className="flex">
      <Sidebar>
        {/* Server Componentのchildrenとして渡す */}
        <NavigationMenu items={navItems} />
      </Sidebar>
      <main>
        <DashboardContent />
      </main>
    </div>
  );
}
```

### 6.4 Server/Client 境界のアンチパターン

```typescript
// ============================================
// アンチパターン1: 不要な 'use client'
// ============================================

// 悪い: 静的なコンポーネントにuse clientを付けている
'use client'; // ← 不要！
function Footer() {
  return (
    <footer>
      <p>2024 My Company. All rights reserved.</p>
    </footer>
  );
}

// 良い: Server Componentとして維持
function Footer() {
  return (
    <footer>
      <p>2024 My Company. All rights reserved.</p>
    </footer>
  );
}

// ============================================
// アンチパターン2: Server Component で関数をpropsに渡す
// ============================================

// 悪い: Server Componentから関数をClient Componentに渡そうとする
async function Page() {
  const handleClick = () => { // ← シリアライズ不可能！
    console.log('clicked');
  };

  return <ClientButton onClick={handleClick} />; // エラー
}

// 良い: Server Actions を使う
async function Page() {
  async function handleSubmit(formData: FormData) {
    'use server';
    const name = formData.get('name');
    await saveUser({ name });
  }

  return (
    <form action={handleSubmit}>
      <input name="name" />
      <button type="submit">保存</button>
    </form>
  );
}

// ============================================
// アンチパターン3: Client境界が高すぎる
// ============================================

// 悪い: レイアウト全体をClientにしてしまう
'use client';
function Layout({ children }) {
  const [theme, setTheme] = useState('light');
  return (
    <div className={theme}>
      <Header />     {/* Server Componentにできるのに */}
      <Sidebar />    {/* Server Componentにできるのに */}
      {children}
      <Footer />     {/* Server Componentにできるのに */}
    </div>
  );
}

// 良い: テーマ切り替えのみをClientに
function Layout({ children }) {
  return (
    <ThemeProvider> {/* ClientのProvider */}
      <Header />     {/* Server Component */}
      <Sidebar />    {/* Server Component */}
      {children}
      <Footer />     {/* Server Component */}
    </ThemeProvider>
  );
}
```

---

## 7. コンポーネントのパフォーマンス最適化

### 7.1 React.memo による再レンダリング最適化

```typescript
// ============================================
// React.memo の適切な使い方
// ============================================

// React.memo を使うべき場面
// → 親が頻繁に再レンダリングされるが、子のpropsは変わらない
// → レンダリングコストが高い（大きなリスト、複雑な計算等）

// --- 例: 高コストのリストアイテム ---
const UserRow = memo(function UserRow({ user, onEdit }: {
  user: User;
  onEdit: (user: User) => void;
}) {
  return (
    <tr>
      <td>
        <div className="flex items-center gap-3">
          <Avatar src={user.avatar} alt={user.name} />
          <div>
            <p className="font-medium">{user.name}</p>
            <p className="text-sm text-gray-500">{user.email}</p>
          </div>
        </div>
      </td>
      <td>
        <Badge variant={user.isActive ? 'success' : 'default'}>
          {user.isActive ? 'アクティブ' : '非アクティブ'}
        </Badge>
      </td>
      <td>
        <Button variant="ghost" size="sm" onClick={() => onEdit(user)}>
          編集
        </Button>
      </td>
    </tr>
  );
});

// 親コンポーネント: onEditをuseCallbackで安定化
function UserTable({ users }: { users: User[] }) {
  const [editingUser, setEditingUser] = useState<User | null>(null);

  // useCallbackで参照を安定化（React.memoの効果を最大化）
  const handleEdit = useCallback((user: User) => {
    setEditingUser(user);
  }, []);

  return (
    <table>
      <tbody>
        {users.map(user => (
          <UserRow key={user.id} user={user} onEdit={handleEdit} />
        ))}
      </tbody>
    </table>
  );
}

// ============================================
// React.memo を使うべきでない場面
// ============================================

// 1. propsが毎回変わるコンポーネント
//    → メモ化のオーバーヘッドが無駄
// 2. レンダリングコストが低いコンポーネント
//    → 比較コスト > レンダリングコスト
// 3. childrenを受け取るコンポーネント
//    → childrenは毎回新しいオブジェクト

// 悪い例: 毎回propsが変わる
const BadMemo = memo(function BadMemo({ items }: { items: Item[] }) {
  return <ul>{items.map(i => <li key={i.id}>{i.name}</li>)}</ul>;
});

// 親: items を毎レンダリングで新しい配列を作成
function Parent() {
  const items = data.filter(d => d.active); // ← 毎回新しい配列
  return <BadMemo items={items} />; // ← memo の効果なし
}

// 良い例: useMemo で配列を安定化
function Parent() {
  const items = useMemo(() => data.filter(d => d.active), [data]);
  return <BadMemo items={items} />; // ← memo が効く
}
```

### 7.2 useMemo と useCallback

```typescript
// ============================================
// useMemo: 計算結果のメモ化
// ============================================

function ExpensiveComponent({ data, filters }: {
  data: DataItem[];
  filters: Filters;
}) {
  // 重い計算をメモ化
  const processedData = useMemo(() => {
    return data
      .filter(item => matchesFilters(item, filters))
      .map(item => transformItem(item))
      .sort((a, b) => b.score - a.score);
  }, [data, filters]);

  // グラフ用の集計データ
  const chartData = useMemo(() => {
    return processedData.reduce((acc, item) => {
      const month = item.date.substring(0, 7);
      acc[month] = (acc[month] || 0) + item.value;
      return acc;
    }, {} as Record<string, number>);
  }, [processedData]);

  return (
    <div>
      <DataChart data={chartData} />
      <DataTable data={processedData} />
    </div>
  );
}

// ============================================
// useCallback: コールバック関数のメモ化
// ============================================

function ParentComponent() {
  const [count, setCount] = useState(0);
  const [items, setItems] = useState<Item[]>([]);

  // setItemsは安定した参照なのでdeps不要
  const handleAddItem = useCallback((item: Item) => {
    setItems(prev => [...prev, item]);
  }, []);

  // 外部の値に依存する場合はdepsに含める
  const handleSubmit = useCallback(async () => {
    await submitItems(items, count);
  }, [items, count]);

  return (
    <div>
      <ItemList items={items} onAdd={handleAddItem} />
      <SubmitButton onSubmit={handleSubmit} />
      <Counter count={count} setCount={setCount} />
    </div>
  );
}
```

### 7.3 コンポーネントの遅延読み込み

```typescript
// ============================================
// React.lazy + Suspense による遅延読み込み
// ============================================

// 通常のimport（バンドルに含まれる）
// import HeavyChart from './HeavyChart';

// 遅延import（必要時にのみ読み込む）
const HeavyChart = lazy(() => import('./HeavyChart'));
const CodeEditor = lazy(() => import('./CodeEditor'));
const MarkdownPreview = lazy(() => import('./MarkdownPreview'));

function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div>
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="overview">概要</TabsTrigger>
          <TabsTrigger value="analytics">分析</TabsTrigger>
          <TabsTrigger value="editor">エディタ</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <OverviewPanel /> {/* 通常読み込み */}
        </TabsContent>

        <TabsContent value="analytics">
          <Suspense fallback={<ChartSkeleton />}>
            <HeavyChart /> {/* 遅延読み込み */}
          </Suspense>
        </TabsContent>

        <TabsContent value="editor">
          <Suspense fallback={<EditorSkeleton />}>
            <CodeEditor /> {/* 遅延読み込み */}
          </Suspense>
        </TabsContent>
      </Tabs>
    </div>
  );
}

// ============================================
// Next.js での dynamic import
// ============================================
import dynamic from 'next/dynamic';

// SSRを無効化して遅延読み込み
const MapComponent = dynamic(() => import('./Map'), {
  ssr: false, // サーバーサイドではレンダリングしない
  loading: () => <MapSkeleton />,
});

// 名前付きエクスポートの遅延読み込み
const BarChart = dynamic(
  () => import('./charts').then(mod => ({ default: mod.BarChart })),
  { loading: () => <ChartSkeleton /> }
);

function LocationPage() {
  return (
    <div>
      <h1>店舗検索</h1>
      <MapComponent locations={locations} />
      <BarChart data={chartData} />
    </div>
  );
}
```

### 7.4 仮想化（Virtualization）

```typescript
// ============================================
// 大量データの仮想スクロール
// ============================================

// @tanstack/react-virtual を使用
import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualizedList({ items }: { items: Item[] }) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 60, // 各アイテムの推定高さ
    overscan: 5, // 画面外に余分にレンダリングする数
  });

  return (
    <div
      ref={parentRef}
      className="h-[600px] overflow-auto"
    >
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map(virtualRow => (
          <div
            key={virtualRow.key}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: `${virtualRow.size}px`,
              transform: `translateY(${virtualRow.start}px)`,
            }}
          >
            <ItemRow item={items[virtualRow.index]} />
          </div>
        ))}
      </div>
    </div>
  );
}

// 10,000件のアイテムでもスムーズに動作
function LargeDataList() {
  const { data: items } = useItems(); // 10,000件

  return (
    <div>
      <p>{items?.length.toLocaleString()} 件のアイテム</p>
      <VirtualizedList items={items ?? []} />
    </div>
  );
}
```

---

## 8. コンポーネントのテスト戦略

### 8.1 テストの種類と使い分け

```
テストピラミッド:

  ┌────────────────┐
  │    E2E テスト    │  → 少数: ユーザーフロー全体
  │   (Playwright)  │
  ├────────────────┤
  │  統合テスト      │  → 中程度: コンポーネント間の連携
  │  (Testing Lib)  │
  ├────────────────┤
  │  単体テスト      │  → 多数: 個別コンポーネント
  │  (Vitest)       │
  └────────────────┘

各テストの役割:
  単体テスト:
  → 個々のコンポーネントを独立してテスト
  → propsを渡してレンダリング結果を検証
  → イベントハンドラの動作を検証
  → 高速、安定、保守容易

  統合テスト:
  → 複数コンポーネントの連携をテスト
  → データ取得 → 表示 → インタラクション
  → APIモック + コンポーネントレンダリング

  E2Eテスト:
  → 実際のブラウザで動作確認
  → ユーザーの操作フロー全体
  → CI/CDパイプラインで実行
```

### 8.2 コンポーネントの単体テスト

```typescript
// ============================================
// Vitest + React Testing Library
// ============================================

import { render, screen, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { Button } from './Button';
import { UserCard } from './UserCard';
import { SearchForm } from './SearchForm';

// --- Button コンポーネントのテスト ---
describe('Button', () => {
  it('デフォルトのvariantでレンダリングされる', () => {
    render(<Button>クリック</Button>);
    const button = screen.getByRole('button', { name: 'クリック' });
    expect(button).toBeInTheDocument();
    expect(button).toHaveClass('bg-primary');
  });

  it('destructive variantが適用される', () => {
    render(<Button variant="destructive">削除</Button>);
    const button = screen.getByRole('button', { name: '削除' });
    expect(button).toHaveClass('bg-destructive');
  });

  it('isLoading時にSpinnerが表示される', () => {
    render(<Button isLoading>保存</Button>);
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
    expect(screen.getByTestId('spinner')).toBeInTheDocument();
  });

  it('クリックイベントが発火する', async () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>クリック</Button>);

    await userEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('disabled時にクリックが無効になる', async () => {
    const handleClick = vi.fn();
    render(<Button disabled onClick={handleClick}>クリック</Button>);

    await userEvent.click(screen.getByRole('button'));
    expect(handleClick).not.toHaveBeenCalled();
  });
});

// --- UserCard コンポーネントのテスト ---
describe('UserCard', () => {
  const mockUser = {
    id: '1',
    name: 'テストユーザー',
    email: 'test@example.com',
    role: 'admin' as const,
    isActive: true,
    avatar: '/avatar.png',
  };

  it('ユーザー情報が正しく表示される', () => {
    render(<UserCard user={mockUser} onEdit={vi.fn()} onDelete={vi.fn()} />);

    expect(screen.getByText('テストユーザー')).toBeInTheDocument();
    expect(screen.getByText('test@example.com')).toBeInTheDocument();
    expect(screen.getByText('admin')).toBeInTheDocument();
  });

  it('アクティブステータスのバッジが表示される', () => {
    render(<UserCard user={mockUser} onEdit={vi.fn()} onDelete={vi.fn()} />);
    expect(screen.getByText('アクティブ')).toBeInTheDocument();
  });

  it('編集ボタンのクリックでonEditが呼ばれる', async () => {
    const handleEdit = vi.fn();
    render(<UserCard user={mockUser} onEdit={handleEdit} onDelete={vi.fn()} />);

    await userEvent.click(screen.getByRole('button', { name: '編集' }));
    expect(handleEdit).toHaveBeenCalledWith(mockUser);
  });
});

// --- SearchForm コンポーネントのテスト ---
describe('SearchForm', () => {
  it('検索クエリを入力して送信できる', async () => {
    const handleSearch = vi.fn();
    render(
      <SearchForm
        onSearch={handleSearch}
        categories={[
          { id: '1', name: 'カテゴリA' },
          { id: '2', name: 'カテゴリB' },
        ]}
      />
    );

    // テキスト入力
    const searchInput = screen.getByPlaceholderText('キーワードを入力...');
    await userEvent.type(searchInput, 'テスト検索');

    // フォーム送信
    await userEvent.click(screen.getByRole('button', { name: '検索' }));

    expect(handleSearch).toHaveBeenCalledWith(
      'テスト検索',
      expect.any(Object)
    );
  });
});
```

### 8.3 Storybookによるコンポーネントドキュメント

```typescript
// ============================================
// Storybook 7+ のCSF3形式
// ============================================

import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './Button';

const meta: Meta<typeof Button> = {
  title: 'UI/Button',
  component: Button,
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['default', 'destructive', 'outline', 'ghost', 'link'],
      description: 'ボタンのスタイルバリアント',
    },
    size: {
      control: 'radio',
      options: ['sm', 'md', 'lg'],
      description: 'ボタンのサイズ',
    },
    isLoading: {
      control: 'boolean',
      description: 'ローディング状態',
    },
    disabled: {
      control: 'boolean',
      description: '無効状態',
    },
  },
};

export default meta;
type Story = StoryObj<typeof Button>;

// デフォルトストーリー
export const Default: Story = {
  args: {
    children: 'ボタン',
  },
};

// バリアント一覧
export const Variants: Story = {
  render: () => (
    <div className="flex gap-4 items-center">
      <Button variant="default">Default</Button>
      <Button variant="destructive">Destructive</Button>
      <Button variant="outline">Outline</Button>
      <Button variant="ghost">Ghost</Button>
      <Button variant="link">Link</Button>
    </div>
  ),
};

// サイズ一覧
export const Sizes: Story = {
  render: () => (
    <div className="flex gap-4 items-center">
      <Button size="sm">Small</Button>
      <Button size="md">Medium</Button>
      <Button size="lg">Large</Button>
    </div>
  ),
};

// ローディング状態
export const Loading: Story = {
  args: {
    isLoading: true,
    children: '保存中...',
  },
};

// アイコン付き
export const WithIcon: Story = {
  render: () => (
    <div className="flex gap-4">
      <Button leftIcon={<PlusIcon className="h-4 w-4" />}>
        追加
      </Button>
      <Button rightIcon={<ArrowRightIcon className="h-4 w-4" />}>
        次へ
      </Button>
      <Button variant="destructive" leftIcon={<TrashIcon className="h-4 w-4" />}>
        削除
      </Button>
    </div>
  ),
};
```

---

## 9. 大規模アプリケーションでのコンポーネント管理

### 9.1 ディレクトリ構成のパターン

```
推奨ディレクトリ構成（Feature-based）:

  src/
  ├── app/                      # Next.js App Router ルーティング
  │   ├── (auth)/               # 認証レイアウトグループ
  │   │   ├── login/
  │   │   └── register/
  │   ├── (dashboard)/          # ダッシュボードレイアウトグループ
  │   │   ├── users/
  │   │   ├── products/
  │   │   └── settings/
  │   └── layout.tsx
  ├── components/
  │   ├── ui/                   # 汎用UIコンポーネント（shadcn/ui等）
  │   │   ├── button.tsx
  │   │   ├── dialog.tsx
  │   │   ├── input.tsx
  │   │   └── ...
  │   ├── layout/               # レイアウトコンポーネント
  │   │   ├── header.tsx
  │   │   ├── sidebar.tsx
  │   │   └── footer.tsx
  │   └── shared/               # 共通ドメインコンポーネント
  │       ├── data-table.tsx
  │       ├── empty-state.tsx
  │       └── page-header.tsx
  ├── features/                 # 機能ごとのモジュール
  │   ├── users/
  │   │   ├── components/       # ユーザー機能のコンポーネント
  │   │   │   ├── user-card.tsx
  │   │   │   ├── user-form.tsx
  │   │   │   └── user-table.tsx
  │   │   ├── hooks/            # ユーザー機能のフック
  │   │   │   ├── use-users.ts
  │   │   │   └── use-user-form.ts
  │   │   ├── api/              # ユーザーAPI
  │   │   │   └── users.ts
  │   │   ├── types/            # ユーザー型定義
  │   │   │   └── user.ts
  │   │   └── index.ts          # 公開API
  │   ├── products/
  │   │   ├── components/
  │   │   ├── hooks/
  │   │   ├── api/
  │   │   └── index.ts
  │   └── auth/
  │       ├── components/
  │       ├── hooks/
  │       └── index.ts
  ├── hooks/                    # グローバルフック
  │   ├── use-debounce.ts
  │   ├── use-media-query.ts
  │   └── use-local-storage.ts
  ├── lib/                      # ユーティリティ
  │   ├── utils.ts
  │   ├── api-client.ts
  │   └── validations.ts
  └── types/                    # グローバル型定義
      └── global.d.ts
```

### 9.2 コンポーネントの命名規則

```
命名規則のベストプラクティス:

  コンポーネント名:
  → PascalCase を使用
  → 具体的な名前にする（Button, UserCard, ProductList）
  → 抽象的な名前を避ける（Wrapper, Container, Manager）
  → ドメイン接頭辞で名前空間を表現（UserCard, UserForm, UserTable）

  ファイル名:
  → kebab-case を推奨（user-card.tsx, product-list.tsx）
  → shadcn/ui スタイル: button.tsx, dialog.tsx
  → 1ファイル1コンポーネントが基本
  → 関連する小さなコンポーネントは同居可能

  Props型名:
  → コンポーネント名 + Props（ButtonProps, UserCardProps）
  → interface を使用（typeでも可だが統一する）

  フック名:
  → use + 動詞/名詞（useUsers, useDebounce, useLocalStorage）
  → カスタムフックは必ず use で始める

  定数・バリアント:
  → camelCase（buttonVariants, alertStyles）
  → UPPER_SNAKE_CASE は設定値のみ（MAX_RETRY_COUNT）
```

### 9.3 コンポーネント設計チェックリスト

```
新しいコンポーネントを作成する際のチェックリスト:

  設計:
  □ 単一責任原則を満たしているか
  □ 適切な粒度レベルか（プリミティブ/複合/ドメイン/ページ）
  □ 既存コンポーネントで代替できないか
  □ 再利用の可能性はあるか

  Props設計:
  □ HTML標準属性を拡張しているか
  □ 必要最小限のpropsか
  □ TypeScriptの型定義が適切か
  □ デフォルト値が設定されているか
  □ Discriminated Unionで型安全か

  アクセシビリティ:
  □ 適切なARIA属性が設定されているか
  □ キーボードナビゲーションに対応しているか
  □ フォーカス管理が適切か
  □ スクリーンリーダーで正しく読み上げられるか
  □ 色のコントラスト比が十分か

  パフォーマンス:
  □ 不要な再レンダリングがないか
  □ React.memoが必要か検討したか
  □ 大きな依存は遅延読み込みしているか
  □ 大量データは仮想化しているか

  テスト:
  □ 単体テストが書かれているか
  □ Storybookストーリーがあるか
  □ エッジケース（空データ、エラー、ローディング）をテストしているか

  Server/Client境界:
  □ Server Componentで十分ではないか
  □ Client境界は最小限か
  □ propsはシリアライズ可能か
```

---

## 10. トラブルシューティング

### 10.1 よくある問題と解決策

```typescript
// ============================================
// 問題1: propsバケツリレー（Prop Drilling）
// ============================================

// 問題: 深い階層にpropsを渡すために中間コンポーネントが不要なpropsを受け取る
function App() {
  const [theme, setTheme] = useState('light');
  return <Layout theme={theme} setTheme={setTheme} />;
}
function Layout({ theme, setTheme }) {
  return <Sidebar theme={theme} setTheme={setTheme} />;
}
function Sidebar({ theme, setTheme }) {
  return <ThemeToggle theme={theme} setTheme={setTheme} />;
}

// 解決策1: Context API
const ThemeContext = createContext<{
  theme: string;
  setTheme: (theme: string) => void;
}>({ theme: 'light', setTheme: () => {} });

function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState('light');
  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// どの階層からでもアクセス可能
function ThemeToggle() {
  const { theme, setTheme } = useContext(ThemeContext);
  return (
    <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
      {theme === 'light' ? 'ダークモード' : 'ライトモード'}
    </button>
  );
}

// 解決策2: コンポジション（childrenを活用）
function Layout() {
  return (
    <div className="flex">
      <Sidebar />
      <main>{/* ... */}</main>
    </div>
  );
}

// ============================================
// 問題2: 不要な再レンダリング
// ============================================

// 問題: Contextの値が変わると全ての消費者が再レンダリング
function AppProvider({ children }) {
  const [user, setUser] = useState(null);
  const [theme, setTheme] = useState('light');
  const [notifications, setNotifications] = useState([]);

  // ← user, theme, notifications のどれが変わっても全消費者が再レンダリング
  return (
    <AppContext.Provider value={{ user, setUser, theme, setTheme, notifications, setNotifications }}>
      {children}
    </AppContext.Provider>
  );
}

// 解決策: Contextを分割する
function UserProvider({ children }) {
  const [user, setUser] = useState(null);
  return (
    <UserContext.Provider value={{ user, setUser }}>
      {children}
    </UserContext.Provider>
  );
}

function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('light');
  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// ============================================
// 問題3: コンポーネントの循環依存
// ============================================

// 問題: A が B をimport し、B が A をimportする
// ComponentA.tsx
// import { ComponentB } from './ComponentB'; // ← 循環！
// ComponentB.tsx
// import { ComponentA } from './ComponentA'; // ← 循環！

// 解決策: 共通部分を別ファイルに抽出
// shared.tsx - 共通の型やユーティリティ
export interface SharedProps { /* ... */ }

// ComponentA.tsx
import { SharedProps } from './shared';
// ComponentAのみの実装

// ComponentB.tsx
import { SharedProps } from './shared';
// ComponentBのみの実装
```

### 10.2 デバッグのテクニック

```typescript
// ============================================
// React DevTools でのコンポーネントデバッグ
// ============================================

// 1. displayName の設定（React DevToolsで識別しやすくする）
const MemoizedComponent = memo(function MyComponent(props) {
  return <div>{/* ... */}</div>;
});
MemoizedComponent.displayName = 'MemoizedComponent';

// forwardRef の場合
const ForwardedInput = forwardRef<HTMLInputElement, InputProps>((props, ref) => {
  return <input ref={ref} {...props} />;
});
ForwardedInput.displayName = 'ForwardedInput';

// 2. useDebugValue でカスタムフックの値をDevToolsに表示
function useOnlineStatus() {
  const [isOnline, setIsOnline] = useState(true);

  useDebugValue(isOnline ? 'オンライン' : 'オフライン');

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return isOnline;
}

// 3. React Profiler で再レンダリングの原因を特定
import { Profiler } from 'react';

function onRender(
  id: string,
  phase: 'mount' | 'update',
  actualDuration: number,
  baseDuration: number,
  startTime: number,
  commitTime: number
) {
  console.log(`[${id}] ${phase}: ${actualDuration.toFixed(2)}ms`);
}

function App() {
  return (
    <Profiler id="UserList" onRender={onRender}>
      <UserList />
    </Profiler>
  );
}

// 4. why-did-you-render ライブラリ
// wdyr.ts
import React from 'react';
if (process.env.NODE_ENV === 'development') {
  const whyDidYouRender = require('@welldone-software/why-did-you-render');
  whyDidYouRender(React, {
    trackAllPureComponents: true,
    logOnDifferentValues: true,
  });
}

// 特定のコンポーネントを追跡
UserList.whyDidYouRender = true;
```

---

## FAQ

### Q1: Atomic DesignとFeature-based構成はどう共存させるか？
Atomic Designはコンポーネントの粒度（Atoms/Molecules/Organisms/Templates/Pages）を定義するための設計思想であり、Feature-basedはプロジェクトのディレクトリ構成のアプローチである。両者は共存可能で、実務では `shared/components/ui/` にAtoms/Molecules相当の汎用UIコンポーネント（Button, Input, Card等）を配置し、`features/xxx/components/` にOrganism相当のドメイン固有コンポーネントを配置するパターンが多い。Atomic Designの5階層を厳密に守る必要はなく、「汎用UI」と「ドメインUI」の2層に簡略化するのが実践的である。

### Q2: Server ComponentとClient Componentの境界はどう決めるか？
基本原則は「Client Componentを可能な限り小さく、末端に配置する」である。データ取得やレンダリングのみのコンポーネントはServer Component、useState/useEffectやイベントハンドラを使うコンポーネントはClient Componentにする。具体的には、ページ全体をServer Componentで構築し、検索入力、モーダル、フォームなどインタラクティブな部分のみを `'use client'` で囲む。こうすることでJavaScriptバンドルサイズが最小化され、初期ロードが高速になる。

### Q3: コンポーネントライブラリ（shadcn/ui等）はカスタマイズすべきか？
shadcn/uiはコピー&ペースト型のUIライブラリであり、プロジェクトに直接コードを取り込むため自由にカスタマイズできる。推奨アプローチは、まずshadcn/uiのデフォルトスタイルをそのまま使い、デザイン要件に合わせて段階的にカスタマイズすることである。Tailwind CSSの設定（`tailwind.config.ts`）でブランドカラーやフォントを調整し、個別コンポーネントのvariantsをcva（class-variance-authority）で管理すると一貫性を保ちやすい。ゼロからUIを構築するよりも、既存ライブラリをベースにカスタマイズする方が圧倒的に効率的である。

---

## まとめ

### コンポーネント設計パターンの全体マップ

| パターン | 用途 | 適用場面 | 難易度 |
|---------|------|---------|--------|
| SRP分割 | 責任の分離 | 全プロジェクト | 低 |
| Atomic Design | UIの階層化 | デザインシステム | 中 |
| Container/Presentational | ロジックとUIの分離 | データ表示コンポーネント | 低 |
| Custom Hooks | ロジックの再利用 | 共通処理の抽出 | 低 |
| Compound Components | 関連UIの宣言的な組み合わせ | Tabs, Accordion等 | 中 |
| Headless UI | ロジック提供、スタイルは自由 | カスタムUI構築 | 中 |
| Variants（cva） | バリアント管理 | デザインシステム | 低 |
| Render Props | 表示のカスタマイズ | 柔軟なUI | 中 |
| Server/Client境界 | Client を最小限に | Next.js App Router | 中 |
| React.memo | 再レンダリング最適化 | パフォーマンス改善 | 中 |
| Virtualization | 大量データ表示 | リスト/テーブル | 高 |
| Lazy Loading | バンドル最適化 | 大きなコンポーネント | 低 |

### 設計判断のフローチャート

```
新しいUIを作る時の判断フロー:

  1. 既存コンポーネントで代替可能？
     → Yes: 既存を使う
     → No: 次へ

  2. UIライブラリ（shadcn/ui等）に同等品がある？
     → Yes: ライブラリを使う
     → No: 次へ

  3. 複数箇所で再利用する？
     → Yes: components/ui/ に汎用コンポーネントを作成
     → No: features/xxx/components/ にドメインコンポーネントを作成

  4. インタラクティブ要素がある？
     → Yes: Client Component（'use client'）
     → No: Server Component のまま

  5. ロジックが複雑？
     → Yes: カスタムフックに抽出
     → No: コンポーネント内にロジックを保持

  6. 大量データを表示する？
     → Yes: 仮想化を検討
     → No: 通常レンダリング
```

---

## 次に読むべきガイド

---

## 参考文献
1. Radix. "Primitives." radix-ui.com, 2024.
2. shadcn/ui. "Re-usable components." ui.shadcn.com, 2024.
3. patterns.dev. "Component Patterns." patterns.dev, 2024.
4. React. "Server Components." react.dev, 2024.
5. Kent C. Dodds. "One React Component Pattern To Rule Them All." kentcdodds.com, 2024.
6. Dan Abramov. "Presentational and Container Components." medium.com, 2015.
7. TanStack. "React Virtual." tanstack.com, 2024.
8. Storybook. "Component Story Format." storybook.js.org, 2024.
9. Joe Bell. "Class Variance Authority." cva.style, 2024.
10. Adobe. "React Aria." react-spectrum.adobe.com, 2024.

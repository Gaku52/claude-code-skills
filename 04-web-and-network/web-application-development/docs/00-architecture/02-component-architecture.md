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

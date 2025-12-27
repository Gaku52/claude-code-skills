# React Component Design 完全ガイド

## 目次
1. [コンポーネント設計原則](#コンポーネント設計原則)
2. [Presentational vs Container](#presentational-vs-container)
3. [Compound Components](#compound-components)
4. [Render Props](#render-props)
5. [Higher-Order Components](#higher-order-components)
6. [コンポーネント構成パターン](#コンポーネント構成パターン)
7. [型安全性](#型安全性)
8. [アクセシビリティ](#アクセシビリティ)

---

## コンポーネント設計原則

### 単一責任の原則（SRP）

```typescript
// ❌ Bad: 1つのコンポーネントが複数の責任を持つ
function UserDashboard() {
  const [user, setUser] = useState(null);
  const [posts, setPosts] = useState([]);
  const [comments, setComments] = useState([]);

  useEffect(() => {
    fetchUser();
    fetchPosts();
    fetchComments();
  }, []);

  return (
    <div>
      <div>{/* User profile */}</div>
      <div>{/* Posts list */}</div>
      <div>{/* Comments list */}</div>
    </div>
  );
}

// ✅ Good: 責任を分離
function UserDashboard() {
  return (
    <div>
      <UserProfile />
      <UserPosts />
      <UserComments />
    </div>
  );
}

function UserProfile() {
  const { data, loading } = useUser();
  if (loading) return <Skeleton />;
  return <div>{data?.name}</div>;
}

function UserPosts() {
  const { data, loading } = usePosts();
  if (loading) return <Skeleton />;
  return <PostList posts={data} />;
}
```

### Props のインターフェース設計

```typescript
// ❌ Bad: 不明確なprops
interface ButtonProps {
  data: any;
  onClick: Function;
  style: any;
}

// ✅ Good: 明確で型安全なprops
interface ButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  className?: string;
  type?: 'button' | 'submit' | 'reset';
}

function Button({
  children,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  onClick,
  className = '',
  type = 'button',
}: ButtonProps) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      onClick={onClick}
      className={`btn btn-${variant} btn-${size} ${className}`}
    >
      {loading ? <Spinner /> : children}
    </button>
  );
}
```

### コンポーネントの粒度

```typescript
// ❌ Bad: 粒度が大きすぎる
function UserCard({ user }: { user: User }) {
  return (
    <div className="card">
      <img src={user.avatar} alt={user.name} />
      <h2>{user.name}</h2>
      <p>{user.email}</p>
      <button onClick={() => follow(user.id)}>Follow</button>
      <button onClick={() => message(user.id)}>Message</button>
      <div>
        <span>Followers: {user.followers}</span>
        <span>Following: {user.following}</span>
      </div>
    </div>
  );
}

// ✅ Good: 適切に分割
function UserCard({ user }: { user: User }) {
  return (
    <Card>
      <UserAvatar src={user.avatar} alt={user.name} />
      <UserInfo name={user.name} email={user.email} />
      <UserActions userId={user.id} />
      <UserStats followers={user.followers} following={user.following} />
    </Card>
  );
}

function UserAvatar({ src, alt }: { src: string; alt: string }) {
  return <img src={src} alt={alt} className="avatar" />;
}

function UserInfo({ name, email }: { name: string; email: string }) {
  return (
    <div className="user-info">
      <h2>{name}</h2>
      <p>{email}</p>
    </div>
  );
}

function UserActions({ userId }: { userId: string }) {
  const { follow } = useFollow();
  const { message } = useMessage();

  return (
    <div className="actions">
      <button onClick={() => follow(userId)}>Follow</button>
      <button onClick={() => message(userId)}>Message</button>
    </div>
  );
}

function UserStats({ followers, following }: { followers: number; following: number }) {
  return (
    <div className="stats">
      <span>Followers: {followers}</span>
      <span>Following: {following}</span>
    </div>
  );
}
```

---

## Presentational vs Container

### Presentational Components

```typescript
// Presentational: UIの見た目のみを担当
interface UserListProps {
  users: User[];
  onUserClick: (user: User) => void;
  loading: boolean;
}

function UserList({ users, onUserClick, loading }: UserListProps) {
  if (loading) {
    return <LoadingSpinner />;
  }

  return (
    <ul className="user-list">
      {users.map((user) => (
        <li key={user.id} onClick={() => onUserClick(user)}>
          <UserCard user={user} />
        </li>
      ))}
    </ul>
  );
}
```

### Container Components

```typescript
// Container: データフェッチとロジックを担当
function UserListContainer() {
  const { data: users, loading } = useFetch<User[]>('/api/users');
  const navigate = useNavigate();

  const handleUserClick = (user: User) => {
    navigate(`/users/${user.id}`);
  };

  return <UserList users={users || []} onUserClick={handleUserClick} loading={loading} />;
}
```

### カスタムフックで分離

```typescript
// ロジックをカスタムフックに抽出
function useUserList() {
  const { data: users, loading, error } = useFetch<User[]>('/api/users');
  const navigate = useNavigate();

  const handleUserClick = useCallback(
    (user: User) => {
      navigate(`/users/${user.id}`);
    },
    [navigate]
  );

  return { users, loading, error, handleUserClick };
}

// Presentational Component
function UserList({ users, onUserClick, loading }: UserListProps) {
  if (loading) return <LoadingSpinner />;
  return (
    <ul>
      {users.map((user) => (
        <UserCard key={user.id} user={user} onClick={() => onUserClick(user)} />
      ))}
    </ul>
  );
}

// Container Component
function UserListContainer() {
  const { users, loading, error, handleUserClick } = useUserList();

  if (error) return <ErrorMessage error={error} />;

  return <UserList users={users || []} onUserClick={handleUserClick} loading={loading} />;
}
```

---

## Compound Components

### 基本的な Compound Components

```typescript
// Compound Component パターン
interface TabsContextValue {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const TabsContext = createContext<TabsContextValue | undefined>(undefined);

function Tabs({ children, defaultTab }: { children: React.ReactNode; defaultTab: string }) {
  const [activeTab, setActiveTab] = useState(defaultTab);

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div className="tabs">{children}</div>
    </TabsContext.Provider>
  );
}

function TabList({ children }: { children: React.ReactNode }) {
  return <div className="tab-list" role="tablist">{children}</div>;
}

function Tab({ value, children }: { value: string; children: React.ReactNode }) {
  const context = useContext(TabsContext);
  if (!context) throw new Error('Tab must be used within Tabs');

  const { activeTab, setActiveTab } = context;
  const isActive = activeTab === value;

  return (
    <button
      role="tab"
      aria-selected={isActive}
      onClick={() => setActiveTab(value)}
      className={isActive ? 'tab active' : 'tab'}
    >
      {children}
    </button>
  );
}

function TabPanel({ value, children }: { value: string; children: React.ReactNode }) {
  const context = useContext(TabsContext);
  if (!context) throw new Error('TabPanel must be used within Tabs');

  const { activeTab } = context;

  if (activeTab !== value) return null;

  return (
    <div role="tabpanel" className="tab-panel">
      {children}
    </div>
  );
}

// 名前空間パターンでエクスポート
Tabs.List = TabList;
Tabs.Tab = Tab;
Tabs.Panel = TabPanel;

// 使用例
function App() {
  return (
    <Tabs defaultTab="profile">
      <Tabs.List>
        <Tabs.Tab value="profile">Profile</Tabs.Tab>
        <Tabs.Tab value="settings">Settings</Tabs.Tab>
        <Tabs.Tab value="notifications">Notifications</Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel value="profile">
        <ProfileContent />
      </Tabs.Panel>
      <Tabs.Panel value="settings">
        <SettingsContent />
      </Tabs.Panel>
      <Tabs.Panel value="notifications">
        <NotificationsContent />
      </Tabs.Panel>
    </Tabs>
  );
}
```

### Select Compound Component

```typescript
interface SelectContextValue {
  value: string;
  onChange: (value: string) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

const SelectContext = createContext<SelectContextValue | undefined>(undefined);

function Select({
  children,
  value,
  onChange,
}: {
  children: React.ReactNode;
  value: string;
  onChange: (value: string) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <SelectContext.Provider value={{ value, onChange, isOpen, setIsOpen }}>
      <div className="select">{children}</div>
    </SelectContext.Provider>
  );
}

function SelectTrigger({ children }: { children: React.ReactNode }) {
  const context = useContext(SelectContext);
  if (!context) throw new Error('SelectTrigger must be used within Select');

  const { isOpen, setIsOpen } = context;

  return (
    <button onClick={() => setIsOpen(!isOpen)} className="select-trigger">
      {children}
    </button>
  );
}

function SelectContent({ children }: { children: React.ReactNode }) {
  const context = useContext(SelectContext);
  if (!context) throw new Error('SelectContent must be used within Select');

  const { isOpen } = context;

  if (!isOpen) return null;

  return <div className="select-content">{children}</div>;
}

function SelectItem({ value, children }: { value: string; children: React.ReactNode }) {
  const context = useContext(SelectContext);
  if (!context) throw new Error('SelectItem must be used within Select');

  const { onChange, setIsOpen } = context;

  return (
    <button
      onClick={() => {
        onChange(value);
        setIsOpen(false);
      }}
      className="select-item"
    >
      {children}
    </button>
  );
}

Select.Trigger = SelectTrigger;
Select.Content = SelectContent;
Select.Item = SelectItem;

// 使用例
function App() {
  const [selectedColor, setSelectedColor] = useState('red');

  return (
    <Select value={selectedColor} onChange={setSelectedColor}>
      <Select.Trigger>
        <span>{selectedColor}</span>
      </Select.Trigger>
      <Select.Content>
        <Select.Item value="red">Red</Select.Item>
        <Select.Item value="green">Green</Select.Item>
        <Select.Item value="blue">Blue</Select.Item>
      </Select.Content>
    </Select>
  );
}
```

---

## Render Props

### 基本的な Render Props

```typescript
interface MousePosition {
  x: number;
  y: number;
}

function Mouse({ render }: { render: (position: MousePosition) => React.ReactNode }) {
  const [position, setPosition] = useState<MousePosition>({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      setPosition({ x: event.clientX, y: event.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return <>{render(position)}</>;
}

// 使用例
function App() {
  return (
    <Mouse
      render={({ x, y }) => (
        <div>
          Mouse position: {x}, {y}
        </div>
      )}
    />
  );
}
```

### Children as Function

```typescript
function DataProvider<T>({
  url,
  children,
}: {
  url: string;
  children: (data: T | null, loading: boolean, error: Error | null) => React.ReactNode;
}) {
  const { data, loading, error } = useFetch<T>(url);

  return <>{children(data, loading, error)}</>;
}

// 使用例
function App() {
  return (
    <DataProvider<User[]> url="/api/users">
      {(users, loading, error) => {
        if (loading) return <div>Loading...</div>;
        if (error) return <div>Error: {error.message}</div>;
        return (
          <ul>
            {users?.map((user) => (
              <li key={user.id}>{user.name}</li>
            ))}
          </ul>
        );
      }}
    </DataProvider>
  );
}
```

---

## Higher-Order Components

### 基本的な HOC

```typescript
function withLoading<P extends object>(
  Component: React.ComponentType<P>
): React.FC<P & { loading: boolean }> {
  return function WithLoadingComponent({ loading, ...props }: P & { loading: boolean }) {
    if (loading) {
      return <LoadingSpinner />;
    }

    return <Component {...(props as P)} />;
  };
}

// 使用例
function UserList({ users }: { users: User[] }) {
  return (
    <ul>
      {users.map((user) => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}

const UserListWithLoading = withLoading(UserList);

function App() {
  const { data, loading } = useFetch<User[]>('/api/users');

  return <UserListWithLoading users={data || []} loading={loading} />;
}
```

### 認証 HOC

```typescript
function withAuth<P extends object>(
  Component: React.ComponentType<P>
): React.FC<P> {
  return function WithAuthComponent(props: P) {
    const { isAuthenticated, user } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
      if (!isAuthenticated) {
        navigate('/login');
      }
    }, [isAuthenticated, navigate]);

    if (!isAuthenticated) {
      return null;
    }

    return <Component {...props} />;
  };
}

// 使用例
const ProtectedDashboard = withAuth(Dashboard);
```

---

## コンポーネント構成パターン

### レイアウトコンポーネント

```typescript
// レイアウトコンポーネント
interface LayoutProps {
  children: React.ReactNode;
  sidebar?: React.ReactNode;
  header?: React.ReactNode;
  footer?: React.ReactNode;
}

function DashboardLayout({ children, sidebar, header, footer }: LayoutProps) {
  return (
    <div className="dashboard-layout">
      {header && <header className="header">{header}</header>}
      <div className="main-content">
        {sidebar && <aside className="sidebar">{sidebar}</aside>}
        <main className="content">{children}</main>
      </div>
      {footer && <footer className="footer">{footer}</footer>}
    </div>
  );
}

// 使用例
function Dashboard() {
  return (
    <DashboardLayout
      header={<Header />}
      sidebar={<Sidebar />}
      footer={<Footer />}
    >
      <DashboardContent />
    </DashboardLayout>
  );
}
```

### Slot Pattern

```typescript
interface CardProps {
  slots: {
    header?: React.ReactNode;
    media?: React.ReactNode;
    content: React.ReactNode;
    actions?: React.ReactNode;
  };
}

function Card({ slots }: CardProps) {
  return (
    <div className="card">
      {slots.header && <div className="card-header">{slots.header}</div>}
      {slots.media && <div className="card-media">{slots.media}</div>}
      <div className="card-content">{slots.content}</div>
      {slots.actions && <div className="card-actions">{slots.actions}</div>}
    </div>
  );
}

// 使用例
function ProductCard({ product }: { product: Product }) {
  return (
    <Card
      slots={{
        header: <h2>{product.name}</h2>,
        media: <img src={product.image} alt={product.name} />,
        content: <p>{product.description}</p>,
        actions: (
          <>
            <button>Add to Cart</button>
            <button>View Details</button>
          </>
        ),
      }}
    />
  );
}
```

---

## 型安全性

### Generics の活用

```typescript
interface ListProps<T> {
  items: T[];
  renderItem: (item: T) => React.ReactNode;
  keyExtractor: (item: T) => string | number;
  emptyMessage?: string;
}

function List<T>({ items, renderItem, keyExtractor, emptyMessage = 'No items' }: ListProps<T>) {
  if (items.length === 0) {
    return <div className="empty">{emptyMessage}</div>;
  }

  return (
    <ul className="list">
      {items.map((item) => (
        <li key={keyExtractor(item)}>{renderItem(item)}</li>
      ))}
    </ul>
  );
}

// 使用例（型推論が効く）
function UserList() {
  const users: User[] = [
    { id: '1', name: 'Alice' },
    { id: '2', name: 'Bob' },
  ];

  return (
    <List
      items={users}
      renderItem={(user) => <UserCard user={user} />}
      keyExtractor={(user) => user.id}
    />
  );
}
```

### Discriminated Unions

```typescript
type ButtonProps =
  | {
      variant: 'link';
      href: string;
      target?: '_blank' | '_self';
      onClick?: never;
    }
  | {
      variant?: 'primary' | 'secondary';
      href?: never;
      target?: never;
      onClick?: () => void;
    };

function Button({ variant = 'primary', href, target, onClick, children }: ButtonProps & { children: React.ReactNode }) {
  if (variant === 'link' && href) {
    return (
      <a href={href} target={target} className="button button-link">
        {children}
      </a>
    );
  }

  return (
    <button onClick={onClick} className={`button button-${variant}`}>
      {children}
    </button>
  );
}

// 使用例（型安全）
<Button variant="link" href="/about">About</Button>  // ✅ OK
<Button variant="primary" onClick={() => {}}>Click</Button>  // ✅ OK
<Button variant="link" onClick={() => {}}>Invalid</Button>  // ❌ Type Error
```

---

## アクセシビリティ

### ARIA属性の適切な使用

```typescript
function Dialog({
  isOpen,
  onClose,
  title,
  children,
}: {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}) {
  const titleId = useId();

  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      return () => document.removeEventListener('keydown', handleEscape);
    }
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      className="dialog-overlay"
      onClick={onClose}
    >
      <div className="dialog-content" onClick={(e) => e.stopPropagation()}>
        <h2 id={titleId}>{title}</h2>
        <button onClick={onClose} aria-label="Close dialog">
          ×
        </button>
        {children}
      </div>
    </div>
  );
}
```

### キーボードナビゲーション

```typescript
function Dropdown({ options, onSelect }: { options: string[]; onSelect: (option: string) => void }) {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const handleKeyDown = (event: React.KeyboardEvent) => {
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setSelectedIndex((prev) => (prev + 1) % options.length);
        break;
      case 'ArrowUp':
        event.preventDefault();
        setSelectedIndex((prev) => (prev - 1 + options.length) % options.length);
        break;
      case 'Enter':
        event.preventDefault();
        onSelect(options[selectedIndex]);
        setIsOpen(false);
        break;
      case 'Escape':
        setIsOpen(false);
        break;
    }
  };

  return (
    <div className="dropdown" onKeyDown={handleKeyDown}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        Select option
      </button>
      {isOpen && (
        <ul role="listbox" tabIndex={-1}>
          {options.map((option, index) => (
            <li
              key={option}
              role="option"
              aria-selected={index === selectedIndex}
              onClick={() => {
                onSelect(option);
                setIsOpen(false);
              }}
            >
              {option}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

---

このガイドでは、Reactコンポーネント設計のベストプラクティスを、設計原則から具体的なパターン（Presentational/Container、Compound Components、Render Props、HOC）、型安全性、アクセシビリティまで包括的に解説しました。これらのパターンを適切に使い分けることで、再利用性が高く保守しやすいコンポーネントを構築できます。

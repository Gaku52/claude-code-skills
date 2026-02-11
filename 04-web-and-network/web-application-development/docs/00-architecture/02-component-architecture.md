# コンポーネント設計

> コンポーネント設計はUIの再利用性と保守性を決定づける。Atomic Design、Container/Presentational、Compound Components、Headless UIまで、スケーラブルなコンポーネントアーキテクチャの全パターンを習得する。

## この章で学ぶこと

- [ ] コンポーネント分割の原則と粒度設計を理解する
- [ ] 主要なコンポーネント設計パターンを把握する
- [ ] Headless UIとコンポーネントライブラリの活用を学ぶ

---

## 1. コンポーネント分割の原則

```
単一責任原則（SRP）:
  → 1コンポーネント = 1つの責任

  悪い例: UserPage が全てを担当
  function UserPage() {
    // データ取得 + フィルタリング + ソート + ページネーション
    // + ユーザー表示 + 編集 + 削除 + 検索 = 500行
  }

  良い例: 責任を分割
  function UserPage() {
    return (
      <PageLayout>
        <UserSearchBar />
        <UserFilters />
        <UserTable />
        <Pagination />
      </PageLayout>
    );
  }

分割の判断基準:
  ✓ 50行を超えたら分割を検討
  ✓ 同じUIパターンが2回以上出現したら共通化
  ✓ テストしたい単位で分割
  ✓ データ取得とUIレンダリングを分離

  分割しすぎの兆候:
  ✗ props が 10個以上のバケツリレー
  ✗ 1つの変更で5ファイル以上修正
  ✗ コンポーネント名が抽象的すぎる
```

---

## 2. Container / Presentational パターン

```
Container（ロジック担当）:
  → データ取得、状態管理、イベントハンドリング
  → UIを持たない（Presentationalに委譲）

Presentational（表示担当）:
  → propsを受け取って表示するだけ
  → 内部状態は最小限（UIの開閉等）
  → テストが容易

// Container
function UserListContainer() {
  const { data: users, isLoading } = useUsers();
  const [filter, setFilter] = useState('all');
  const filtered = users?.filter(u =>
    filter === 'all' ? true : u.role === filter
  );

  return (
    <UserListView
      users={filtered ?? []}
      isLoading={isLoading}
      filter={filter}
      onFilterChange={setFilter}
    />
  );
}

// Presentational
function UserListView({ users, isLoading, filter, onFilterChange }) {
  if (isLoading) return <Skeleton />;
  return (
    <div>
      <FilterBar value={filter} onChange={onFilterChange} />
      <ul>
        {users.map(user => (
          <UserCard key={user.id} user={user} />
        ))}
      </ul>
    </div>
  );
}

現代的なアプローチ:
  → React Server Components で自然に分離
  → Server Component = Container
  → Client Component = Presentational
```

---

## 3. Compound Components

```typescript
// Compound Components: 関連コンポーネントの組み合わせ
// 例: Tabs コンポーネント

// 使い方（宣言的で直感的）
<Tabs defaultValue="profile">
  <Tabs.List>
    <Tabs.Trigger value="profile">Profile</Tabs.Trigger>
    <Tabs.Trigger value="settings">Settings</Tabs.Trigger>
    <Tabs.Trigger value="billing">Billing</Tabs.Trigger>
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

// 実装
const TabsContext = createContext<{
  activeTab: string;
  setActiveTab: (value: string) => void;
} | null>(null);

function Tabs({ defaultValue, children }: {
  defaultValue: string;
  children: ReactNode;
}) {
  const [activeTab, setActiveTab] = useState(defaultValue);
  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div role="tablist">{children}</div>
    </TabsContext.Provider>
  );
}

Tabs.List = function TabsList({ children }: { children: ReactNode }) {
  return <div className="flex gap-2 border-b">{children}</div>;
};

Tabs.Trigger = function TabsTrigger({ value, children }: {
  value: string; children: ReactNode;
}) {
  const ctx = useContext(TabsContext)!;
  return (
    <button
      role="tab"
      aria-selected={ctx.activeTab === value}
      onClick={() => ctx.setActiveTab(value)}
      className={ctx.activeTab === value ? 'border-b-2 font-bold' : ''}
    >
      {children}
    </button>
  );
};

Tabs.Content = function TabsContent({ value, children }: {
  value: string; children: ReactNode;
}) {
  const ctx = useContext(TabsContext)!;
  if (ctx.activeTab !== value) return null;
  return <div role="tabpanel">{children}</div>;
};
```

---

## 4. Headless UI

```
Headless UI:
  → ロジック・状態管理・アクセシビリティのみ提供
  → スタイルは利用者が自由に定義
  → 最大限の柔軟性

ライブラリ:
  Radix UI:     最も人気、shadcn/ui のベース
  Headless UI:  Tailwind Labs 製
  React Aria:   Adobe 製、アクセシビリティ最高
  Ariakit:      軽量、コンポーザブル

shadcn/ui の仕組み:
  → Radix UI（Headless）+ Tailwind CSS（スタイル）
  → コピー＆ペーストでコンポーネントを追加
  → 完全にカスタマイズ可能
  → node_modules に依存しない

  npx shadcn-ui@latest add button
  → src/components/ui/button.tsx が生成される
  → 中身を自由に編集可能

コンポーネントライブラリの選定:
  フルスタイル:
  → MUI: Material Design、機能豊富
  → Ant Design: エンタープライズ向け
  → Chakra UI: DX重視

  Headless + スタイル:
  → shadcn/ui: Radix + Tailwind（推奨）
  → Ark UI: 新しい、フレームワーク非依存

  Headless のみ:
  → Radix UI
  → React Aria
```

---

## 5. Props設計

```typescript
// 良い Props 設計の原則

// ① オブジェクトではなく個別の props（浅い場合）
// 悪い: <UserCard user={user} />
// 良い: <UserCard name={user.name} email={user.email} avatar={user.avatar} />
// → ただしフィールドが多い場合はオブジェクトで渡す

// ② children を活用（コンポジション）
// 悪い
<Card title="Users" body={<UserList />} footer={<Pagination />} />

// 良い
<Card>
  <Card.Header>Users</Card.Header>
  <Card.Body><UserList /></Card.Body>
  <Card.Footer><Pagination /></Card.Footer>
</Card>

// ③ Render Props / Slots
<DataTable
  data={users}
  columns={columns}
  renderRow={(user) => <UserRow user={user} />}
  renderEmpty={() => <EmptyState message="No users found" />}
/>

// ④ バリアント（cva / class-variance-authority）
import { cva, type VariantProps } from 'class-variance-authority';

const buttonVariants = cva(
  'inline-flex items-center rounded-md font-medium transition-colors',
  {
    variants: {
      variant: {
        default: 'bg-primary text-white hover:bg-primary/90',
        outline: 'border border-input hover:bg-accent',
        ghost: 'hover:bg-accent',
        destructive: 'bg-red-500 text-white hover:bg-red-600',
      },
      size: {
        sm: 'h-8 px-3 text-sm',
        md: 'h-10 px-4',
        lg: 'h-12 px-6 text-lg',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'md',
    },
  }
);

interface ButtonProps extends
  React.ButtonHTMLAttributes<HTMLButtonElement>,
  VariantProps<typeof buttonVariants> {}

function Button({ variant, size, className, ...props }: ButtonProps) {
  return <button className={buttonVariants({ variant, size, className })} {...props} />;
}

// <Button variant="destructive" size="lg">Delete</Button>
```

---

## 6. Server / Client コンポーネント境界

```
Next.js App Router でのコンポーネント設計:

  ルール:
  → デフォルトは Server Component
  → 'use client' は必要最小限に
  → Client の境界をなるべく葉（リーフ）に近づける

  // ✓ 良い: Client境界が小さい
  // page.tsx (Server)
  async function ProductPage({ params }) {
    const product = await getProduct(params.id);
    return (
      <div>
        <h1>{product.name}</h1>
        <p>{product.description}</p>
        <ProductPrice price={product.price} />
        <AddToCartButton productId={product.id} />  {/* Client */}
      </div>
    );
  }

  // ✗ 悪い: ページ全体がClient
  'use client';
  function ProductPage({ params }) {
    const { data: product } = useQuery(...);
    // ...全てクライアントサイド
  }

判断基準:
  Server Component を使う:
  → データベースアクセス
  → サーバーのみのAPI呼び出し
  → 大きな依存（マークダウンパーサー等）
  → 機密情報の処理

  Client Component を使う:
  → useState, useEffect が必要
  → onClick 等のイベントハンドラ
  → ブラウザAPI（localStorage等）
  → サードパーティのクライアントライブラリ
```

---

## まとめ

| パターン | 用途 |
|---------|------|
| Container/Presentational | ロジックとUIの分離 |
| Compound Components | 関連UIの宣言的な組み合わせ |
| Headless UI | ロジック提供、スタイルは自由 |
| Variants（cva） | バリアント管理 |
| Server/Client境界 | Client を最小限に |

---

## 次に読むべきガイド
→ [[03-data-fetching-patterns.md]] — データフェッチング

---

## 参考文献
1. Radix. "Primitives." radix-ui.com, 2024.
2. shadcn/ui. "Re-usable components." ui.shadcn.com, 2024.
3. patterns.dev. "Component Patterns." patterns.dev, 2024.

---
name: react-development
description: React開発の詳細ガイド。Hooks、コンポーネント設計、パフォーマンス最適化、テストなど、Reactアプリケーション開発のベストプラクティス。
---

# React Development Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [Hooks活用](#hooks活用)
4. [コンポーネント設計](#コンポーネント設計)
5. [パフォーマンス最適化](#パフォーマンス最適化)
6. [実践例](#実践例)
7. [アンチパターン](#アンチパターン)
8. [Agent連携](#agent連携)

---

## 概要

このSkillは、React開発の詳細をカバーします：

- **Hooks** - useState, useEffect, カスタムフック
- **コンポーネント設計** - 再利用可能なコンポーネント
- **パフォーマンス最適化** - memo, useMemo, useCallback
- **状態管理** - Context API, 外部ライブラリ
- **フォーム処理** - react-hook-form
- **テスト** - React Testing Library

---

## いつ使うか

### 🎯 必須のタイミング

- [ ] Reactコンポーネント作成時
- [ ] カスタムフック作成時
- [ ] パフォーマンス問題発生時
- [ ] フォーム実装時

---

## Hooks活用

### 基本Hooks

#### useState - 状態管理

```tsx
// ✅ 基本的な使用
function Counter() {
  const [count, setCount] = useState(0)

  return (
    <button onClick={() => setCount(count + 1)}>
      Count: {count}
    </button>
  )
}

// ✅ 複雑な状態（オブジェクト）
function UserForm() {
  const [form, setForm] = useState({
    name: '',
    email: ''
  })

  const handleChange = (field: string, value: string) => {
    setForm(prev => ({ ...prev, [field]: value }))
  }

  return (
    <>
      <input
        value={form.name}
        onChange={(e) => handleChange('name', e.target.value)}
      />
      <input
        value={form.email}
        onChange={(e) => handleChange('email', e.target.value)}
      />
    </>
  )
}
```

#### useEffect - 副作用

```tsx
// ✅ データフェッチ
function UserProfile({ userId }: { userId: string }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let ignore = false

    async function fetchUser() {
      setLoading(true)
      const data = await fetch(`/api/users/${userId}`).then(r => r.json())

      if (!ignore) {
        setUser(data)
        setLoading(false)
      }
    }

    fetchUser()

    return () => {
      ignore = true // クリーンアップ
    }
  }, [userId])

  if (loading) return <div>Loading...</div>
  if (!user) return <div>Not found</div>

  return <div>{user.name}</div>
}
```

#### useRef - DOM参照

```tsx
function SearchInput() {
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    // マウント時にフォーカス
    inputRef.current?.focus()
  }, [])

  return <input ref={inputRef} placeholder="Search..." />
}
```

### カスタムHooks

#### データフェッチフック

```tsx
// hooks/useFetch.ts
function useFetch<T>(url: string) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    let ignore = false

    async function fetchData() {
      try {
        setLoading(true)
        const response = await fetch(url)
        if (!response.ok) throw new Error('Failed to fetch')
        const json = await response.json()

        if (!ignore) {
          setData(json)
          setError(null)
        }
      } catch (e) {
        if (!ignore) {
          setError(e as Error)
        }
      } finally {
        if (!ignore) {
          setLoading(false)
        }
      }
    }

    fetchData()

    return () => {
      ignore = true
    }
  }, [url])

  return { data, loading, error }
}

// 使用例
function UserList() {
  const { data: users, loading, error } = useFetch<User[]>('/api/users')

  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error.message}</div>

  return (
    <ul>
      {users?.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  )
}
```

#### ローカルストレージフック

```tsx
// hooks/useLocalStorage.ts
function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : initialValue
    } catch (error) {
      console.error(error)
      return initialValue
    }
  })

  const setValue = (value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value
      setStoredValue(valueToStore)
      window.localStorage.setItem(key, JSON.stringify(valueToStore))
    } catch (error) {
      console.error(error)
    }
  }

  return [storedValue, setValue] as const
}

// 使用例
function App() {
  const [theme, setTheme] = useLocalStorage<'light' | 'dark'>('theme', 'light')

  return (
    <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
      Current: {theme}
    </button>
  )
}
```

---

## コンポーネント設計

### 再利用可能なボタン

```tsx
// components/ui/Button.tsx
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  isLoading?: boolean
}

export function Button({
  variant = 'primary',
  size = 'md',
  isLoading = false,
  children,
  className,
  disabled,
  ...props
}: ButtonProps) {
  const baseStyles = 'rounded font-medium transition-colors'

  const variants = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300',
    danger: 'bg-red-600 text-white hover:bg-red-700'
  }

  const sizes = {
    sm: 'px-3 py-1 text-sm',
    md: 'px-4 py-2',
    lg: 'px-6 py-3 text-lg'
  }

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? 'Loading...' : children}
    </button>
  )
}

// 使用例
<Button variant="primary" size="lg" onClick={handleSubmit}>
  Submit
</Button>
```

### コンパウンドコンポーネント

```tsx
// components/Tabs.tsx
interface TabsContextValue {
  activeTab: string
  setActiveTab: (tab: string) => void
}

const TabsContext = React.createContext<TabsContextValue | undefined>(undefined)

function Tabs({ children, defaultTab }: { children: React.ReactNode; defaultTab: string }) {
  const [activeTab, setActiveTab] = useState(defaultTab)

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      {children}
    </TabsContext.Provider>
  )
}

function TabList({ children }: { children: React.ReactNode }) {
  return <div className="flex gap-2 border-b">{children}</div>
}

function Tab({ value, children }: { value: string; children: React.ReactNode }) {
  const context = React.useContext(TabsContext)
  if (!context) throw new Error('Tab must be used within Tabs')

  const { activeTab, setActiveTab } = context

  return (
    <button
      className={activeTab === value ? 'border-b-2 border-blue-600' : ''}
      onClick={() => setActiveTab(value)}
    >
      {children}
    </button>
  )
}

function TabPanel({ value, children }: { value: string; children: React.ReactNode }) {
  const context = React.useContext(TabsContext)
  if (!context) throw new Error('TabPanel must be used within Tabs')

  const { activeTab } = context
  if (activeTab !== value) return null

  return <div>{children}</div>
}

Tabs.List = TabList
Tabs.Tab = Tab
Tabs.Panel = TabPanel

// 使用例
<Tabs defaultTab="profile">
  <Tabs.List>
    <Tabs.Tab value="profile">Profile</Tabs.Tab>
    <Tabs.Tab value="settings">Settings</Tabs.Tab>
  </Tabs.List>

  <Tabs.Panel value="profile">
    <p>Profile content</p>
  </Tabs.Panel>
  <Tabs.Panel value="settings">
    <p>Settings content</p>
  </Tabs.Panel>
</Tabs>
```

---

## パフォーマンス最適化

### React.memo - 不要な再レンダリング防止

```tsx
// ❌ 悪い例（毎回再レンダリング）
function UserCard({ user }: { user: User }) {
  console.log('Rendering UserCard')
  return <div>{user.name}</div>
}

// ✅ 良い例（propsが変わったときのみ再レンダリング）
const UserCard = React.memo(({ user }: { user: User }) => {
  console.log('Rendering UserCard')
  return <div>{user.name}</div>
})
```

### useMemo - 高コストな計算のメモ化

```tsx
function ExpensiveList({ items, filter }: { items: Item[]; filter: string }) {
  // ✅ filter または items が変わったときのみ再計算
  const filteredItems = useMemo(() => {
    console.log('Filtering items...')
    return items.filter(item => item.name.includes(filter))
  }, [items, filter])

  return (
    <ul>
      {filteredItems.map(item => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  )
}
```

### useCallback - 関数のメモ化

```tsx
function TodoList() {
  const [todos, setTodos] = useState<Todo[]>([])

  // ✅ 関数をメモ化（子コンポーネントに渡す場合に重要）
  const handleToggle = useCallback((id: string) => {
    setTodos(prev =>
      prev.map(todo =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
      )
    )
  }, [])

  return (
    <ul>
      {todos.map(todo => (
        <TodoItem key={todo.id} todo={todo} onToggle={handleToggle} />
      ))}
    </ul>
  )
}

const TodoItem = React.memo(({ todo, onToggle }: {
  todo: Todo;
  onToggle: (id: string) => void
}) => {
  return (
    <li>
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={() => onToggle(todo.id)}
      />
      {todo.title}
    </li>
  )
})
```

---

## アンチパターン

### ❌ 1. useEffectの無限ループ

```tsx
// ❌ 悪い例
function BadComponent() {
  const [data, setData] = useState([])

  useEffect(() => {
    fetch('/api/data')
      .then(res => res.json())
      .then(setData) // dataが更新 → useEffectが再実行 → 無限ループ
  }, [data])
}

// ✅ 良い例
function GoodComponent() {
  const [data, setData] = useState([])

  useEffect(() => {
    fetch('/api/data')
      .then(res => res.json())
      .then(setData)
  }, []) // 依存配列が空 → マウント時のみ実行
}
```

### ❌ 2. 過剰なuseCallback/useMemo

```tsx
// ❌ 悪い例（不要なメモ化）
function Component() {
  const name = useMemo(() => 'John', []) // 不要
  const greet = useCallback(() => console.log('Hello'), []) // 不要

  return <div>{name}</div>
}

// ✅ 良い例
function Component() {
  const name = 'John'
  const greet = () => console.log('Hello')

  return <div>{name}</div>
}
```

---

## Agent連携

### 📖 Agentへの指示例

**カスタムフック作成**
```
データフェッチ用のカスタムフックuseFetchを作成してください。
loading、error、dataを返すようにしてください。
```

**コンポーネント作成**
```
ユーザーカードコンポーネントを作成してください。
- ユーザー名、メール、アバターを表示
- hover時にシャドウを表示
- クリック時に詳細ページに遷移
```

---

## まとめ

### Reactのベストプラクティス

1. **Hooksを活用** - 状態管理、副作用、カスタムフック
2. **コンポーネント設計** - 再利用可能、単一責任
3. **パフォーマンス最適化** - memo, useMemo, useCallback
4. **型安全性** - TypeScript活用

---

_Last updated: 2025-12-24_

# React トラブルシューティング

## 目次

1. [概要](#概要)
2. [セットアップ・環境構築エラー](#セットアップ環境構築エラー)
3. [Hooks関連エラー](#hooks関連エラー)
4. [コンポーネントエラー](#コンポーネントエラー)
5. [State・Props関連エラー](#stateprops関連エラー)
6. [イベント処理エラー](#イベント処理エラー)
7. [ルーティングエラー](#ルーティングエラー)
8. [ビルド・デプロイエラー](#ビルドデプロイエラー)
9. [パフォーマンス問題](#パフォーマンス問題)

---

## 概要

このガイドは、React開発で頻繁に遭遇するエラーと解決策をまとめたトラブルシューティングデータベースです。

**収録エラー数:** 25個

**対象バージョン:** React 18.x

---

## セットアップ・環境構築エラー

### ❌ エラー1: Cannot find module 'react'

```
Error: Cannot find module 'react'
```

**原因:**
- `node_modules`が削除された
- `package.json`の依存関係が未インストール

**解決策:**

```bash
# 依存関係を再インストール
npm install

# または
npm install react react-dom
```

---

### ❌ エラー2: Module not found: Can't resolve 'react-router-dom'

```
Module not found: Error: Can't resolve 'react-router-dom'
```

**原因:**
- `react-router-dom`がインストールされていない

**解決策:**

```bash
npm install react-router-dom
```

---

### ❌ エラー3: You are running `create-react-app` with an old version

```
You are running `create-react-app` 4.0.3, which is behind the latest release (5.0.1).
We no longer support global installation of Create React App.
```

**原因:**
- グローバルにインストールされた古い`create-react-app`

**解決策:**

```bash
# 古いバージョンをアンインストール
npm uninstall -g create-react-app

# 最新版で作成
npx create-react-app my-app

# または Vite を使用（推奨）
npm create vite@latest my-app -- --template react-ts
```

---

### ❌ エラー4: Port 3000 is already in use

```
Something is already running on port 3000.
```

**原因:**
- ポート3000が既に使用中

**解決策:**

```bash
# macOS/Linux
lsof -ti :3000 | xargs kill -9

# Windows (PowerShell)
Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess | Stop-Process

# または別のポートを使用
PORT=3001 npm start
```

---

## Hooks関連エラー

### ❌ エラー5: Hooks can only be called inside the body of a function component

```
Error: Invalid hook call. Hooks can only be called inside of the body of a function component.
```

**原因:**
- クラスコンポーネント内でHooksを使用
- 条件分岐内でHooksを使用
- 通常の関数内でHooksを使用

**間違った例:**

```typescript
// ❌ 条件分岐内でHooksを使用
function MyComponent() {
  if (condition) {
    const [count, setCount] = useState(0) // エラー
  }
  return <div>Test</div>
}

// ❌ 通常の関数内でHooksを使用
function regularFunction() {
  const [count, setCount] = useState(0) // エラー
}
```

**解決策:**

```typescript
// ✅ 正しい使用方法
function MyComponent() {
  const [count, setCount] = useState(0)

  if (condition) {
    // Hooksの結果を使用
  }

  return <div>{count}</div>
}
```

---

### ❌ エラー6: Cannot update a component while rendering a different component

```
Warning: Cannot update a component (`Parent`) while rendering a different component (`Child`).
```

**原因:**
- レンダリング中にstateを更新している

**間違った例:**

```typescript
// ❌ レンダリング中にstateを更新
function Parent() {
  const [count, setCount] = useState(0)

  return (
    <Child onRender={() => setCount(count + 1)} />
  )
}

function Child({ onRender }) {
  onRender() // レンダリング中に実行される
  return <div>Child</div>
}
```

**解決策:**

```typescript
// ✅ useEffectで更新
function Parent() {
  const [count, setCount] = useState(0)

  return (
    <Child onMount={() => setCount(count + 1)} />
  )
}

function Child({ onMount }) {
  useEffect(() => {
    onMount()
  }, [onMount])

  return <div>Child</div>
}
```

---

### ❌ エラー7: Too many re-renders

```
Error: Too many re-renders. React limits the number of renders to prevent an infinite loop.
```

**原因:**
- useEffectの依存配列が不適切
- stateの更新が無限ループを引き起こしている

**間違った例:**

```typescript
// ❌ 無限ループ
function MyComponent() {
  const [count, setCount] = useState(0)

  useEffect(() => {
    setCount(count + 1) // 依存配列なし → 無限ループ
  })

  return <div>{count}</div>
}
```

**解決策:**

```typescript
// ✅ 適切な依存配列
function MyComponent() {
  const [count, setCount] = useState(0)

  useEffect(() => {
    setCount(count + 1)
  }, []) // 空配列 → 初回のみ実行

  return <div>{count}</div>
}

// ✅ または条件を追加
function MyComponent() {
  const [count, setCount] = useState(0)

  useEffect(() => {
    if (count < 10) {
      setCount(count + 1)
    }
  }, [count])

  return <div>{count}</div>
}
```

---

### ❌ エラー8: useEffect has a missing dependency

```
React Hook useEffect has a missing dependency: 'fetchData'.
Either include it or remove the dependency array.
```

**原因:**
- useEffect内で使用している変数が依存配列に含まれていない

**間違った例:**

```typescript
// ❌ fetchDataが依存配列にない
function MyComponent({ userId }) {
  const [data, setData] = useState(null)

  const fetchData = async () => {
    const result = await fetch(`/api/users/${userId}`)
    setData(result)
  }

  useEffect(() => {
    fetchData()
  }, []) // 警告: fetchData, userId が依存配列にない

  return <div>{data}</div>
}
```

**解決策:**

```typescript
// ✅ 方法1: useCallbackで関数をメモ化
function MyComponent({ userId }) {
  const [data, setData] = useState(null)

  const fetchData = useCallback(async () => {
    const result = await fetch(`/api/users/${userId}`)
    setData(result)
  }, [userId])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  return <div>{data}</div>
}

// ✅ 方法2: useEffect内で定義
function MyComponent({ userId }) {
  const [data, setData] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      const result = await fetch(`/api/users/${userId}`)
      setData(result)
    }
    fetchData()
  }, [userId])

  return <div>{data}</div>
}
```

---

## コンポーネントエラー

### ❌ エラー9: Objects are not valid as a React child

```
Error: Objects are not valid as a React child (found: object with keys {name, age}).
```

**原因:**
- JSX内でオブジェクトを直接レンダリングしようとしている

**間違った例:**

```typescript
// ❌ オブジェクトを直接レンダリング
function MyComponent() {
  const user = { name: '太郎', age: 25 }
  return <div>{user}</div> // エラー
}
```

**解決策:**

```typescript
// ✅ プロパティを個別にレンダリング
function MyComponent() {
  const user = { name: '太郎', age: 25 }
  return (
    <div>
      <p>名前: {user.name}</p>
      <p>年齢: {user.age}</p>
    </div>
  )
}

// ✅ JSON.stringifyで表示（デバッグ用）
function MyComponent() {
  const user = { name: '太郎', age: 25 }
  return <div>{JSON.stringify(user)}</div>
}
```

---

### ❌ エラー10: Each child in a list should have a unique "key" prop

```
Warning: Each child in a list should have a unique "key" prop.
```

**原因:**
- リストをレンダリングする際にkey propが設定されていない

**間違った例:**

```typescript
// ❌ keyがない
function UserList({ users }) {
  return (
    <ul>
      {users.map(user => (
        <li>{user.name}</li> // 警告
      ))}
    </ul>
  )
}
```

**解決策:**

```typescript
// ✅ 一意のIDをkeyに設定
function UserList({ users }) {
  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  )
}

// ⚠️ 最終手段: indexをkeyに（非推奨）
function UserList({ users }) {
  return (
    <ul>
      {users.map((user, index) => (
        <li key={index}>{user.name}</li>
      ))}
    </ul>
  )
}
```

**注意:** indexをkeyに使うのは、リストの順序が変わらない場合のみにしてください。

---

### ❌ エラー11: Component definition is missing display name

```
Component definition is missing display name
```

**原因:**
- ESLintのルールで、無名関数コンポーネントが禁止されている

**間違った例:**

```typescript
// ❌ 無名関数
export default () => {
  return <div>Test</div>
}
```

**解決策:**

```typescript
// ✅ 名前付き関数
export default function MyComponent() {
  return <div>Test</div>
}

// ✅ または変数に代入
const MyComponent = () => {
  return <div>Test</div>
}

MyComponent.displayName = 'MyComponent'

export default MyComponent
```

---

## State・Props関連エラー

### ❌ エラー12: Cannot read property 'map' of undefined

```
TypeError: Cannot read property 'map' of undefined
```

**原因:**
- 配列が`undefined`または`null`の状態で`map`を呼び出している

**間違った例:**

```typescript
// ❌ 初期値がundefined
function UserList() {
  const [users, setUsers] = useState()

  return (
    <ul>
      {users.map(user => ( // エラー
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  )
}
```

**解決策:**

```typescript
// ✅ 初期値を空配列に設定
function UserList() {
  const [users, setUsers] = useState([])

  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  )
}

// ✅ オプショナルチェイニング + デフォルト値
function UserList({ users }) {
  return (
    <ul>
      {(users ?? []).map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  )
}
```

---

### ❌ エラー13: Cannot update during an existing state transition

```
Warning: Cannot update during an existing state transition (such as within `render`).
```

**原因:**
- レンダリング中にstateを更新している

**間違った例:**

```typescript
// ❌ レンダリング中にsetStateを呼び出し
function MyComponent() {
  const [count, setCount] = useState(0)

  setCount(count + 1) // エラー

  return <div>{count}</div>
}
```

**解決策:**

```typescript
// ✅ useEffectで更新
function MyComponent() {
  const [count, setCount] = useState(0)

  useEffect(() => {
    setCount(count + 1)
  }, [])

  return <div>{count}</div>
}

// ✅ イベントハンドラーで更新
function MyComponent() {
  const [count, setCount] = useState(0)

  return (
    <div>
      <p>{count}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
    </div>
  )
}
```

---

### ❌ エラー14: setState is not a function

```
TypeError: setCount is not a function
```

**原因:**
- useStateの戻り値を正しく分割代入していない

**間違った例:**

```typescript
// ❌ 分割代入が間違っている
function MyComponent() {
  const [count] = useState(0) // setCountがない

  const handleClick = () => {
    setCount(count + 1) // エラー
  }

  return <button onClick={handleClick}>{count}</button>
}
```

**解決策:**

```typescript
// ✅ 正しい分割代入
function MyComponent() {
  const [count, setCount] = useState(0)

  const handleClick = () => {
    setCount(count + 1)
  }

  return <button onClick={handleClick}>{count}</button>
}
```

---

## イベント処理エラー

### ❌ エラー15: onClick関数が即座に実行される

**問題:**
- ボタンをクリックする前に関数が実行される

**間違った例:**

```typescript
// ❌ 関数を即座に実行
function MyComponent() {
  const handleClick = () => {
    alert('Clicked!')
  }

  return <button onClick={handleClick()}>Click</button> // ()付き
}
```

**解決策:**

```typescript
// ✅ 関数参照を渡す
function MyComponent() {
  const handleClick = () => {
    alert('Clicked!')
  }

  return <button onClick={handleClick}>Click</button>
}

// ✅ 引数を渡す場合はアロー関数
function MyComponent() {
  const handleClick = (id) => {
    alert(`Clicked: ${id}`)
  }

  return <button onClick={() => handleClick(1)}>Click</button>
}
```

---

### ❌ エラー16: Event.preventDefault() is not a function

```
TypeError: e.preventDefault is not a function
```

**原因:**
- イベントオブジェクトが渡されていない

**間違った例:**

```typescript
// ❌ イベントオブジェクトがない
function MyForm() {
  const handleSubmit = () => {
    e.preventDefault() // エラー: eが定義されていない
  }

  return <form onSubmit={handleSubmit}>...</form>
}
```

**解決策:**

```typescript
// ✅ イベントオブジェクトを引数で受け取る
function MyForm() {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // フォーム処理
  }

  return <form onSubmit={handleSubmit}>...</form>
}
```

---

## ルーティングエラー

### ❌ エラー17: useNavigate may be used only in the context of a Router

```
Error: useNavigate() may be used only in the context of a <Router> component.
```

**原因:**
- `useNavigate`を`<BrowserRouter>`の外で使用している

**間違った例:**

```typescript
// ❌ BrowserRouterの外でuseNavigateを使用
function App() {
  return (
    <div>
      <LoginButton /> {/* エラー */}
    </div>
  )
}

function LoginButton() {
  const navigate = useNavigate() // エラー
  return <button onClick={() => navigate('/login')}>Login</button>
}
```

**解決策:**

```typescript
// ✅ BrowserRouterで囲む
function App() {
  return (
    <BrowserRouter>
      <div>
        <LoginButton />
      </div>
    </BrowserRouter>
  )
}

function LoginButton() {
  const navigate = useNavigate()
  return <button onClick={() => navigate('/login')}>Login</button>
}
```

---

### ❌ エラー18: No routes matched location

```
No routes matched location "/about"
```

**原因:**
- ルートが定義されていない

**間違った例:**

```typescript
// ❌ /aboutルートが定義されていない
<Routes>
  <Route path="/" element={<Home />} />
  <Route path="/contact" element={<Contact />} />
</Routes>
```

**解決策:**

```typescript
// ✅ ルートを追加
<Routes>
  <Route path="/" element={<Home />} />
  <Route path="/about" element={<About />} />
  <Route path="/contact" element={<Contact />} />
</Routes>

// ✅ 404ページを追加
<Routes>
  <Route path="/" element={<Home />} />
  <Route path="/about" element={<About />} />
  <Route path="*" element={<NotFound />} />
</Routes>
```

---

## ビルド・デプロイエラー

### ❌ エラー19: Module parse failed: Unexpected token

```
Module parse failed: Unexpected token (1:0)
You may need an appropriate loader to handle this file type.
```

**原因:**
- TypeScriptファイルをJavaScriptとして読み込もうとしている
- Webpackの設定が不足

**解決策:**

```bash
# TypeScript設定を確認
npm install --save-dev typescript @types/react @types/react-dom

# tsconfig.jsonを作成
npx tsc --init
```

**tsconfig.json:**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true
  },
  "include": ["src"]
}
```

---

### ❌ エラー20: ELIFECYCLE Command failed

```
npm ERR! code ELIFECYCLE
npm ERR! errno 1
```

**原因:**
- ビルドスクリプトの実行に失敗
- ESLintエラー、TypeScriptエラーが存在

**解決策:**

```bash
# 詳細なエラーログを確認
npm run build -- --verbose

# ESLintエラーを無視してビルド
ESLINT_NO_DEV_ERRORS=true npm run build

# TypeScriptエラーを無視（非推奨）
DISABLE_ESLINT_PLUGIN=true npm run build
```

**根本解決:**
- エラーを1つずつ修正する
- `npm run lint`でエラーを確認

---

### ❌ エラー21: Failed to compile (chunk too large)

```
WARNING in asset size limit: The following asset(s) exceed the recommended size limit (244 KiB).
```

**原因:**
- バンドルサイズが大きすぎる

**解決策:**

```typescript
// ✅ コード分割（React.lazy）
const Dashboard = React.lazy(() => import('./Dashboard'))
const Settings = React.lazy(() => import('./Settings'))

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Suspense>
  )
}
```

**vite.config.ts（Vite使用時）:**

```typescript
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom'],
        },
      },
    },
  },
})
```

---

## パフォーマンス問題

### ❌ エラー22: コンポーネントが何度も再レンダリングされる

**原因:**
- 不必要な再レンダリング
- メモ化されていない関数・オブジェクト

**間違った例:**

```typescript
// ❌ 毎回新しいオブジェクトを作成
function Parent() {
  const config = { theme: 'dark' } // 毎レンダリングで新しいオブジェクト

  return <Child config={config} />
}
```

**解決策:**

```typescript
// ✅ useMemoでメモ化
function Parent() {
  const config = useMemo(() => ({ theme: 'dark' }), [])

  return <Child config={config} />
}

// ✅ React.memoでコンポーネントをメモ化
const Child = React.memo(({ config }) => {
  console.log('Child rendered')
  return <div>{config.theme}</div>
})
```

---

### ❌ エラー23: メモリリーク警告

```
Warning: Can't perform a React state update on an unmounted component.
```

**原因:**
- アンマウント後にstateを更新しようとしている

**間違った例:**

```typescript
// ❌ クリーンアップがない
function MyComponent() {
  const [data, setData] = useState(null)

  useEffect(() => {
    fetchData().then(result => setData(result)) // アンマウント後も実行される
  }, [])

  return <div>{data}</div>
}
```

**解決策:**

```typescript
// ✅ クリーンアップ関数を追加
function MyComponent() {
  const [data, setData] = useState(null)

  useEffect(() => {
    let isMounted = true

    fetchData().then(result => {
      if (isMounted) {
        setData(result)
      }
    })

    return () => {
      isMounted = false
    }
  }, [])

  return <div>{data}</div>
}

// ✅ AbortControllerを使用（fetch API）
function MyComponent() {
  const [data, setData] = useState(null)

  useEffect(() => {
    const controller = new AbortController()

    fetch('/api/data', { signal: controller.signal })
      .then(res => res.json())
      .then(setData)
      .catch(err => {
        if (err.name !== 'AbortError') {
          console.error(err)
        }
      })

    return () => controller.abort()
  }, [])

  return <div>{data}</div>
}
```

---

### ❌ エラー24: 画像の読み込みが遅い

**原因:**
- 画像サイズが最適化されていない
- Lazy loadingが実装されていない

**解決策:**

```typescript
// ✅ Lazy loading
<img
  src="/large-image.jpg"
  loading="lazy"
  alt="Description"
/>

// ✅ React Suspenseで画像読み込み
function ImageComponent() {
  return (
    <Suspense fallback={<Skeleton />}>
      <img src="/large-image.jpg" alt="Description" />
    </Suspense>
  )
}

// ✅ 画像最適化ライブラリ
import Image from 'next/image' // Next.js

<Image
  src="/image.jpg"
  width={500}
  height={300}
  alt="Description"
/>
```

---

### ❌ エラー25: フォーム入力が遅い（制御コンポーネント）

**原因:**
- 入力ごとに親コンポーネント全体が再レンダリングされる

**間違った例:**

```typescript
// ❌ 親コンポーネントの再レンダリング
function ParentForm() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  })

  return (
    <div>
      <HeavyComponent /> {/* 入力ごとに再レンダリング */}
      <input
        value={formData.name}
        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
      />
      {/* 他のフィールド */}
    </div>
  )
}
```

**解決策:**

```typescript
// ✅ フォームを別コンポーネントに分離
function ParentForm() {
  return (
    <div>
      <HeavyComponent /> {/* 再レンダリングされない */}
      <FormFields />
    </div>
  )
}

const FormFields = React.memo(() => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  })

  return (
    <form>
      <input
        value={formData.name}
        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
      />
      {/* 他のフィールド */}
    </form>
  )
})

// ✅ 非制御コンポーネント（useRef）
function UncontrolledForm() {
  const nameRef = useRef<HTMLInputElement>(null)

  const handleSubmit = () => {
    console.log(nameRef.current?.value)
  }

  return (
    <form onSubmit={handleSubmit}>
      <input ref={nameRef} />
    </form>
  )
}
```

---

## まとめ

### このガイドで学んだこと

- ✅ React開発における25の頻出エラー
- ✅ 各エラーの原因と解決策
- ✅ ベストプラクティス

### エラー解決の基本手順

1. **エラーメッセージを読む** - 最初の行に重要な情報
2. **スタックトレースを確認** - エラーが発生したファイル・行番号
3. **公式ドキュメントを確認** - [React Docs](https://react.dev/)
4. **このガイドで検索** - よくあるエラーはここに記載
5. **Google/Stack Overflow** - 同じエラーに遭遇した人の解決策

### さらに学ぶ

- **[React公式ドキュメント](https://react.dev/)**
- **[React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)**
- **[React DevTools](https://react.dev/learn/react-developer-tools)** - デバッグツール

---

**関連ガイド:**
- [React Development - 基礎ガイド](../react-development/SKILL.md)
- [統合プロジェクト - フルスタックアプリ](../integrated-projects/fullstack-task-app/)

**親ガイド:** [トラブルシューティングDB](./README.md)

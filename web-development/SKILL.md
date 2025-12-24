---
name: web-development
description: モダンWeb開発の基礎。React、Vue、Next.jsなどのフレームワーク選定、プロジェクト構成、状態管理、ルーティング、ビルドツールなど、Web開発全般のベストプラクティス。
---

# Web Development Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [フレームワーク選定](#フレームワーク選定)
4. [プロジェクト構成](#プロジェクト構成)
5. [開発環境](#開発環境)
6. [実践例](#実践例)
7. [アンチパターン](#アンチパターン)
8. [Agent連携](#agent連携)

---

## 概要

このSkillは、モダンWeb開発の基礎をカバーします：

- **フレームワーク選定** - React, Vue, Next.js, Remix等
- **プロジェクト構成** - ディレクトリ構造、ファイル命名規則
- **状態管理** - Context API, Redux, Zustand, Jotai
- **ルーティング** - React Router, Next.js App Router
- **ビルドツール** - Vite, Webpack, Turbopack
- **CSS戦略** - Tailwind CSS, CSS Modules, Styled Components

---

## いつ使うか

### 🎯 必須のタイミング

- [ ] 新規Webプロジェクト開始時（フレームワーク選定）
- [ ] プロジェクト構成設計時
- [ ] 状態管理戦略決定時
- [ ] CSS戦略決定時

### 🔄 定期的に

- [ ] 依存関係更新時
- [ ] パフォーマンス最適化時
- [ ] 新機能追加時

---

## フレームワーク選定

### 選定基準

| フレームワーク | 用途 | メリット | デメリット |
|----------|-----|--------|---------|
| **Next.js** | フルスタックWebアプリ | SSR/SSG標準、SEO最適、App Router | 学習コスト高 |
| **React (Vite)** | SPA、管理画面 | シンプル、柔軟 | SEO対策が必要 |
| **Remix** | フルスタック | ネストルーティング、UX最高 | エコシステム小 |
| **Vue (Nuxt)** | フルスタック | 学習容易、日本語情報豊富 | React比でエコシステム小 |
| **Astro** | コンテンツサイト | 超高速、部分的インタラクティブ | 複雑なアプリには不向き |

### 判断フローチャート

```
SEOが最重要？
├─ Yes → サーバーサイドレンダリング必要
│   ├─ React好き → Next.js
│   └─ Vue好き → Nuxt.js
└─ No → SPA OK
    ├─ 管理画面・内部ツール → React + Vite
    └─ コンテンツ中心 → Astro
```

---

## プロジェクト構成

### Next.js App Router（推奨）

```
project/
├── app/                    # アプリケーションルート
│   ├── (marketing)/       # ルートグループ（パスに含まれない）
│   │   ├── layout.tsx
│   │   ├── page.tsx       # /
│   │   └── about/page.tsx # /about
│   ├── dashboard/
│   │   ├── layout.tsx
│   │   └── page.tsx       # /dashboard
│   ├── api/               # APIルート
│   │   └── users/route.ts # /api/users
│   ├── layout.tsx         # ルートレイアウト
│   └── globals.css
├── components/            # 共有コンポーネント
│   ├── ui/               # UIコンポーネント（shadcn/ui等）
│   │   ├── button.tsx
│   │   └── card.tsx
│   └── features/         # 機能別コンポーネント
│       ├── user-profile.tsx
│       └── post-list.tsx
├── lib/                   # ユーティリティ、ヘルパー
│   ├── utils.ts
│   ├── api.ts
│   └── db.ts
├── hooks/                 # カスタムフック
│   ├── use-user.ts
│   └── use-posts.ts
├── types/                 # TypeScript型定義
│   ├── user.ts
│   └── post.ts
├── public/                # 静的ファイル
│   └── images/
├── .env.local             # 環境変数
├── next.config.js
├── tailwind.config.ts
└── package.json
```

### React + Vite（SPA）

```
project/
├── src/
│   ├── pages/            # ページコンポーネント
│   │   ├── Home.tsx
│   │   ├── Dashboard.tsx
│   │   └── Settings.tsx
│   ├── components/       # 共有コンポーネント
│   │   ├── Header.tsx
│   │   ├── Sidebar.tsx
│   │   └── ui/
│   ├── hooks/            # カスタムフック
│   ├── lib/              # ユーティリティ
│   ├── store/            # 状態管理（Zustand等）
│   │   ├── userStore.ts
│   │   └── appStore.ts
│   ├── types/            # 型定義
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── public/
├── index.html
├── vite.config.ts
└── package.json
```

---

## 開発環境

### 推奨ツールセット

#### パッケージマネージャー
- **pnpm** (推奨) - 高速、ディスク効率的
- npm, yarn - 標準的

#### フォーマット・Lint
```json
{
  "devDependencies": {
    "eslint": "^8.0.0",
    "prettier": "^3.0.0",
    "typescript": "^5.0.0"
  }
}
```

**.prettierrc**
```json
{
  "semi": false,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5"
}
```

**.eslintrc.json**
```json
{
  "extends": [
    "next/core-web-vitals",
    "plugin:@typescript-eslint/recommended",
    "prettier"
  ],
  "rules": {
    "@typescript-eslint/no-unused-vars": "error",
    "prefer-const": "error"
  }
}
```

#### Git Hooks（Husky）

```bash
pnpm add -D husky lint-staged

# package.json
{
  "lint-staged": {
    "*.{ts,tsx}": ["eslint --fix", "prettier --write"]
  }
}
```

---

## 実践例

### Example 1: Next.js新規プロジェクト（App Router）

```bash
# プロジェクト作成
pnpm create next-app@latest my-app --typescript --tailwind --app

cd my-app

# 追加パッケージ
pnpm add zustand zod react-hook-form
pnpm add -D @types/node

# shadcn/ui セットアップ
pnpm dlx shadcn-ui@latest init
pnpm dlx shadcn-ui@latest add button card
```

**app/page.tsx**
```tsx
import { Button } from '@/components/ui/button'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <h1 className="text-4xl font-bold">Welcome</h1>
      <Button className="mt-4">Get Started</Button>
    </main>
  )
}
```

### Example 2: React + Vite（SPA）

```bash
# プロジェクト作成
pnpm create vite@latest my-app --template react-ts

cd my-app
pnpm install

# 追加パッケージ
pnpm add react-router-dom zustand
pnpm add -D tailwindcss postcss autoprefixer
pnpm dlx tailwindcss init -p
```

**src/main.tsx**
```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
)
```

### Example 3: 状態管理（Zustand）

```typescript
// store/userStore.ts
import { create } from 'zustand'

interface User {
  id: string
  name: string
  email: string
}

interface UserStore {
  user: User | null
  setUser: (user: User) => void
  logout: () => void
}

export const useUserStore = create<UserStore>((set) => ({
  user: null,
  setUser: (user) => set({ user }),
  logout: () => set({ user: null }),
}))
```

**使用例**
```tsx
import { useUserStore } from '@/store/userStore'

export function UserProfile() {
  const { user, logout } = useUserStore()

  if (!user) return <div>Not logged in</div>

  return (
    <div>
      <p>{user.name}</p>
      <button onClick={logout}>Logout</button>
    </div>
  )
}
```

---

## アンチパターン

### ❌ 1. 過度なコンポーネント分割

```tsx
// ❌ 悪い例
function UserName({ name }: { name: string }) {
  return <span>{name}</span>
}

function UserEmail({ email }: { email: string }) {
  return <span>{email}</span>
}

function UserProfile({ user }: { user: User }) {
  return (
    <div>
      <UserName name={user.name} />
      <UserEmail email={user.email} />
    </div>
  )
}
```

```tsx
// ✅ 良い例
function UserProfile({ user }: { user: User }) {
  return (
    <div>
      <span>{user.name}</span>
      <span>{user.email}</span>
    </div>
  )
}
```

### ❌ 2. Prop Drilling

```tsx
// ❌ 悪い例
function App() {
  const [user, setUser] = useState<User | null>(null)
  return <Dashboard user={user} setUser={setUser} />
}

function Dashboard({ user, setUser }: Props) {
  return <Settings user={user} setUser={setUser} />
}

function Settings({ user, setUser }: Props) {
  // userとsetUserを使う
}
```

```tsx
// ✅ 良い例（Zustand使用）
const useUserStore = create<UserStore>((set) => ({
  user: null,
  setUser: (user) => set({ user }),
}))

function Settings() {
  const { user, setUser } = useUserStore()
  // 直接アクセス
}
```

### ❌ 3. useEffectの誤用

```tsx
// ❌ 悪い例
function UserList() {
  const [users, setUsers] = useState<User[]>([])

  useEffect(() => {
    fetch('/api/users')
      .then(res => res.json())
      .then(setUsers)
  }, []) // 依存配列が空 → コンポーネントマウント時のみ
}
```

```tsx
// ✅ 良い例（Next.js App Router）
async function UserList() {
  const users = await fetch('/api/users').then(res => res.json())

  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  )
}
```

---

## Agent連携

### 📖 Agentへの指示例

**新規プロジェクト作成**
```
Next.js App Routerプロジェクトを作成してください。
以下を含めてください：
- TypeScript
- Tailwind CSS
- shadcn/ui
- Zustand（状態管理）
- Prisma（データベース）
```

**コンポーネント生成**
```
ユーザープロフィール編集コンポーネントを作成してください。
以下の機能を含めてください：
- react-hook-formでバリデーション
- zodスキーマ
- 送信時の楽観的UI更新
```

**状態管理の追加**
```
Zustandでショッピングカート機能を実装してください。
addItem, removeItem, clearCartアクションを含めてください。
```

### 🤖 Agentからの提案例

```
このコンポーネントは複雑になっています。
以下に分割することを提案します：

1. UserProfileForm - フォーム部分
2. UserAvatar - アバター表示部分
3. UserActions - アクション部分

分割しますか？
```

---

## まとめ

### Web開発のベストプラクティス

1. **フレームワーク選定** - 用途に応じた最適な選択
2. **プロジェクト構成** - スケーラブルな構造
3. **状態管理** - 適切な粒度で管理
4. **型安全性** - TypeScriptを活用

### 次のステップ

- [ ] フレームワーク選定
- [ ] プロジェクト構成設計
- [ ] 状態管理戦略決定
- [ ] CSS戦略決定

---

## 関連Skills

- **nextjs-development** - Next.js特化ガイド
- **react-development** - React詳細ガイド
- **frontend-performance** - パフォーマンス最適化
- **web-accessibility** - アクセシビリティ対応

---

_Last updated: 2025-12-24_

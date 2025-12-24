---
name: nextjs-development
description: Next.js App Router開発ガイド。Server Components、ルーティング、データフェッチング、キャッシング、デプロイなど、Next.js開発のベストプラクティス。
---

# Next.js Development Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [App Router基礎](#app-router基礎)
4. [Server Components vs Client Components](#server-components-vs-client-components)
5. [データフェッチング](#データフェッチング)
6. [キャッシング戦略](#キャッシング戦略)
7. [実践例](#実践例)
8. [アンチパターン](#アンチパターン)
9. [Agent連携](#agent連携)

---

## 概要

このSkillは、Next.js App Router開発をカバーします：

- **App Router** - ファイルベースルーティング
- **Server Components** - サーバーサイドレンダリング
- **データフェッチング** - fetch, Prisma, ORMs
- **キャッシング** - 自動キャッシュ、revalidate
- **API Routes** - RESTful API
- **デプロイ** - Vercel, 自己ホスティング

---

## いつ使うか

### 🎯 必須のタイミング

- [ ] 新規Next.jsプロジェクト作成時
- [ ] ページ・レイアウト追加時
- [ ] API Route追加時
- [ ] データフェッチング実装時

---

## App Router基礎

### ファイルベースルーティング

```
app/
├── page.tsx                  # / （ルート）
├── about/page.tsx            # /about
├── blog/
│   ├── page.tsx              # /blog
│   └── [slug]/page.tsx       # /blog/hello-world
├── dashboard/
│   ├── layout.tsx            # /dashboard のレイアウト
│   ├── page.tsx              # /dashboard
│   └── settings/page.tsx     # /dashboard/settings
└── api/
    └── users/route.ts        # /api/users
```

### ページの作成

```tsx
// app/page.tsx（ルートページ）
export default function Home() {
  return (
    <main>
      <h1>Welcome</h1>
    </main>
  )
}
```

### レイアウトの作成

```tsx
// app/layout.tsx（ルートレイアウト）
export const metadata = {
  title: 'My App',
  description: 'App description',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ja">
      <body>
        <nav>ナビゲーション</nav>
        {children}
        <footer>フッター</footer>
      </body>
    </html>
  )
}

// app/dashboard/layout.tsx（ネストレイアウト）
export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex">
      <aside>サイドバー</aside>
      <main>{children}</main>
    </div>
  )
}
```

### 動的ルート

```tsx
// app/blog/[slug]/page.tsx
interface PageProps {
  params: { slug: string }
  searchParams: { [key: string]: string | string[] | undefined }
}

export default function BlogPost({ params }: PageProps) {
  return <h1>Post: {params.slug}</h1>
}

// 静的生成用
export async function generateStaticParams() {
  const posts = await getPosts()

  return posts.map((post) => ({
    slug: post.slug,
  }))
}
```

---

## Server Components vs Client Components

### Server Components（デフォルト）

```tsx
// app/posts/page.tsx
// ✅ Server Component（デフォルト）

async function getPosts() {
  const res = await fetch('https://api.example.com/posts', {
    next: { revalidate: 3600 } // 1時間キャッシュ
  })
  return res.json()
}

export default async function PostsPage() {
  const posts = await getPosts() // 直接await可能

  return (
    <ul>
      {posts.map(post => (
        <li key={post.id}>{post.title}</li>
      ))}
    </ul>
  )
}
```

**メリット：**
- サーバーで実行（クライアントバンドル削減）
- 直接DBアクセス可能
- 環境変数を安全に使用可能

### Client Components

```tsx
// components/Counter.tsx
'use client' // ← 必須

import { useState } from 'react'

export function Counter() {
  const [count, setCount] = useState(0)

  return (
    <button onClick={() => setCount(count + 1)}>
      Count: {count}
    </button>
  )
}
```

**使用するタイミング：**
- useState, useEffect等のHooksを使う
- イベントハンドラー（onClick等）
- ブラウザAPI（localStorage等）

### 混在パターン

```tsx
// app/page.tsx（Server Component）
import { Counter } from '@/components/Counter' // Client Component

async function getInitialCount() {
  // サーバーでデータ取得
  return 42
}

export default async function Home() {
  const initialCount = await getInitialCount()

  return (
    <div>
      <h1>Server Component</h1>
      <Counter initialValue={initialCount} />
    </div>
  )
}
```

---

## データフェッチング

### fetch API

```tsx
// キャッシュあり（デフォルト）
async function getData() {
  const res = await fetch('https://api.example.com/data')
  return res.json()
}

// キャッシュなし
async function getData() {
  const res = await fetch('https://api.example.com/data', {
    cache: 'no-store'
  })
  return res.json()
}

// 時間ベースリバリデーション
async function getData() {
  const res = await fetch('https://api.example.com/data', {
    next: { revalidate: 3600 } // 1時間
  })
  return res.json()
}
```

### Prisma使用例

```tsx
// lib/prisma.ts
import { PrismaClient } from '@prisma/client'

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined
}

export const prisma = globalForPrisma.prisma ?? new PrismaClient()

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma

// app/users/page.tsx
import { prisma } from '@/lib/prisma'

export default async function UsersPage() {
  const users = await prisma.user.findMany()

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

## キャッシング戦略

### リバリデーション

#### 時間ベース（Time-based）

```tsx
// 60秒ごとに再検証
fetch('https://api.example.com/data', {
  next: { revalidate: 60 }
})
```

#### オンデマンド（On-demand）

```tsx
// app/api/revalidate/route.ts
import { revalidatePath } from 'next/cache'
import { NextRequest } from 'next/server'

export async function POST(request: NextRequest) {
  const path = request.nextUrl.searchParams.get('path')

  if (path) {
    revalidatePath(path)
    return Response.json({ revalidated: true, now: Date.now() })
  }

  return Response.json({ revalidated: false })
}

// 使用例
// POST /api/revalidate?path=/posts
```

---

## 実践例

### Example 1: ブログアプリ

```tsx
// app/blog/page.tsx
import Link from 'next/link'

async function getPosts() {
  const res = await fetch('https://jsonplaceholder.typicode.com/posts', {
    next: { revalidate: 3600 }
  })
  return res.json()
}

export default async function BlogPage() {
  const posts = await getPosts()

  return (
    <div>
      <h1>Blog</h1>
      <ul>
        {posts.map((post: any) => (
          <li key={post.id}>
            <Link href={`/blog/${post.id}`}>
              {post.title}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}

// app/blog/[id]/page.tsx
async function getPost(id: string) {
  const res = await fetch(`https://jsonplaceholder.typicode.com/posts/${id}`, {
    next: { revalidate: 3600 }
  })
  return res.json()
}

export default async function PostPage({ params }: { params: { id: string } }) {
  const post = await getPost(params.id)

  return (
    <article>
      <h1>{post.title}</h1>
      <p>{post.body}</p>
    </article>
  )
}
```

### Example 2: API Route（CRUD）

```tsx
// app/api/users/route.ts
import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

// GET /api/users
export async function GET() {
  const users = await prisma.user.findMany()
  return NextResponse.json(users)
}

// POST /api/users
export async function POST(request: NextRequest) {
  const body = await request.json()

  const user = await prisma.user.create({
    data: {
      name: body.name,
      email: body.email,
    },
  })

  return NextResponse.json(user, { status: 201 })
}

// app/api/users/[id]/route.ts
// PUT /api/users/:id
export async function PUT(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const body = await request.json()

  const user = await prisma.user.update({
    where: { id: params.id },
    data: body,
  })

  return NextResponse.json(user)
}

// DELETE /api/users/:id
export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  await prisma.user.delete({
    where: { id: params.id },
  })

  return new NextResponse(null, { status: 204 })
}
```

### Example 3: フォーム送信（Server Actions）

```tsx
// app/create-post/page.tsx
import { redirect } from 'next/navigation'
import { prisma } from '@/lib/prisma'

async function createPost(formData: FormData) {
  'use server' // Server Action

  const title = formData.get('title') as string
  const content = formData.get('content') as string

  await prisma.post.create({
    data: { title, content },
  })

  redirect('/posts')
}

export default function CreatePostPage() {
  return (
    <form action={createPost}>
      <input name="title" placeholder="Title" required />
      <textarea name="content" placeholder="Content" required />
      <button type="submit">Create</button>
    </form>
  )
}
```

---

## アンチパターン

### ❌ 1. Client ComponentでのDB直接アクセス

```tsx
'use client'
// ❌ 悪い例
import { prisma } from '@/lib/prisma'

export function UserList() {
  const users = await prisma.user.findMany() // エラー！
}
```

```tsx
// ✅ 良い例（Server Component）
import { prisma } from '@/lib/prisma'

export default async function UserList() {
  const users = await prisma.user.findMany()
  return <ul>{/* ... */}</ul>
}
```

### ❌ 2. 不要な'use client'

```tsx
// ❌ 悪い例
'use client' // 不要（インタラクティブでない）

export function UserCard({ user }: { user: User }) {
  return <div>{user.name}</div>
}
```

```tsx
// ✅ 良い例（Server Component）
export function UserCard({ user }: { user: User }) {
  return <div>{user.name}</div>
}
```

---

## Agent連携

### 📖 Agentへの指示例

**新規ページ作成**
```
/about ページを作成してください。
会社概要、ミッション、チーム紹介を含めてください。
```

**API Route作成**
```
/api/posts のCRUD APIを作成してください。
Prismaを使用して、GET, POST, PUT, DELETEをサポートしてください。
```

**Server Actions実装**
```
ユーザー作成フォームをServer Actionsで実装してください。
バリデーションも含めてください。
```

---

## まとめ

### Next.jsのベストプラクティス

1. **Server Components優先** - デフォルトで使用
2. **適切なキャッシング** - revalidateを活用
3. **型安全性** - TypeScript + Prisma
4. **Server Actions** - フォーム送信に活用

---

_Last updated: 2025-12-24_

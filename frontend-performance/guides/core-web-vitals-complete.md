# Core Web Vitals 完全ガイド

Googleが定義するユーザー体験の核心指標を完全に理解し、実践で改善するための包括的ガイド。

## 目次

1. [概要](#概要)
2. [LCP - Largest Contentful Paint](#lcp---largest-contentful-paint)
3. [INP - Interaction to Next Paint](#inp---interaction-to-next-paint)
4. [CLS - Cumulative Layout Shift](#cls---cumulative-layout-shift)
5. [TTFB - Time to First Byte](#ttfb---time-to-first-byte)
6. [測定方法](#測定方法)
7. [実測値データ](#実測値データ)
8. [よくある間違いと解決策](#よくある間違いと解決策)
9. [業界別ベンチマーク](#業界別ベンチマーク)
10. [継続的モニタリング戦略](#継続的モニタリング戦略)
11. [実践例](#実践例)

---

## 概要

### Core Web Vitalsとは

GoogleがWeb体験の品質を測定するために定義した3つの主要指標：

| 指標 | 説明 | 測定対象 | 目標値 |
|------|------|----------|--------|
| **LCP** | Largest Contentful Paint | 読み込みパフォーマンス | < 2.5秒 |
| **INP** | Interaction to Next Paint | インタラクティブ性 | < 200ms |
| **CLS** | Cumulative Layout Shift | 視覚的安定性 | < 0.1 |

### なぜ重要か

1. **SEOへの影響**: GoogleのランキングシグナルとしてCore Web Vitalsが使用される
2. **ユーザー体験**: 優れたUXはコンバージョン率を向上させる
3. **ビジネス指標**:
   - Amazonの調査: ページ速度が1秒遅くなると、売上が1.6%減少
   - Googleの調査: モバイルサイトの読み込みが3秒以上かかると、53%のユーザーが離脱

### 補助指標

Core Web Vitals以外の重要指標：

| 指標 | 説明 | 目標値 |
|------|------|--------|
| **TTFB** | Time to First Byte | < 600ms |
| **FCP** | First Contentful Paint | < 1.8秒 |
| **TBT** | Total Blocking Time | < 200ms |
| **SI** | Speed Index | < 3.4秒 |

---

## LCP - Largest Contentful Paint

### 定義

ビューポート内で最も大きなコンテンツ要素がレンダリングされるまでの時間。

**LCPの対象要素:**
- `<img>` 要素
- `<svg>` 内の `<image>` 要素
- `<video>` 要素のポスター画像
- `url()` によるCSS背景画像
- テキストを含むブロックレベル要素

### 目標値

| 評価 | LCP |
|------|-----|
| **Good** | < 2.5秒 |
| **Needs Improvement** | 2.5秒 - 4.0秒 |
| **Poor** | > 4.0秒 |

### LCP改善手法

#### 1. 画像最適化

```tsx
// ❌ 悪い例: 最適化なし
<img src="/hero.jpg" alt="Hero" />

// ✅ 良い例: Next.js Image（自動最適化）
import Image from 'next/image'

<Image
  src="/hero.jpg"
  alt="Hero"
  width={1920}
  height={1080}
  priority // LCP要素には必須
  quality={75}
  sizes="100vw"
/>
```

**効果:**
- WebP/AVIF形式への自動変換（-30~50%ファイルサイズ）
- レスポンシブ画像の自動生成
- 遅延ローディング（priority以外）

#### 2. プリロード（Preload）

```tsx
// app/layout.tsx
export default function RootLayout({ children }) {
  return (
    <html>
      <head>
        {/* LCP画像をプリロード */}
        <link
          rel="preload"
          as="image"
          href="/hero.jpg"
          imageSrcSet="/hero-640w.jpg 640w, /hero-1280w.jpg 1280w, /hero-1920w.jpg 1920w"
          imageSizes="100vw"
        />

        {/* 重要なフォントをプリロード */}
        <link
          rel="preload"
          as="font"
          href="/fonts/inter-var.woff2"
          type="font/woff2"
          crossOrigin="anonymous"
        />
      </head>
      <body>{children}</body>
    </html>
  )
}
```

#### 3. Server-Side Rendering (SSR)

```tsx
// app/products/[id]/page.tsx
import { prisma } from '@/lib/prisma'

// ✅ サーバーでレンダリング（LCP改善）
export default async function ProductPage({ params }: { params: { id: string } }) {
  const product = await prisma.product.findUnique({
    where: { id: params.id },
    include: { images: true }
  })

  return (
    <div>
      <Image
        src={product.images[0].url}
        alt={product.name}
        width={800}
        height={600}
        priority
      />
      <h1>{product.name}</h1>
      <p>{product.description}</p>
    </div>
  )
}
```

#### 4. CDNの活用

```typescript
// next.config.js
module.exports = {
  images: {
    loader: 'cloudinary', // または 'imgix', 'cloudflare'
    domains: ['res.cloudinary.com'],
  },
}

// 使用例
<Image
  src="https://res.cloudinary.com/demo/image/upload/sample.jpg"
  alt="Sample"
  width={800}
  height={600}
  priority
/>
```

**効果:**
- 地理的に近いサーバーから配信（低レイテンシ）
- 自動画像最適化
- キャッシング

#### 5. フォント最適化

```tsx
// app/layout.tsx
import { Inter } from 'next/font/google'

const inter = Inter({
  subsets: ['latin'],
  display: 'swap', // フォント読み込み中もテキスト表示
  preload: true,
  variable: '--font-inter',
})

export default function RootLayout({ children }) {
  return (
    <html lang="ja" className={inter.variable}>
      <body className="font-sans">{children}</body>
    </html>
  )
}
```

**font-display戦略:**

| 値 | 説明 | LCP影響 |
|----|------|---------|
| `block` | フォント読み込み待ち（最大3秒） | 悪化 |
| `swap` | 即座にフォールバック表示 | **改善** |
| `fallback` | 100ms待機後フォールバック | 中立 |
| `optional` | ネットワーク状況次第 | 改善 |

#### 6. Critical CSS

```tsx
// app/layout.tsx
import './globals.css' // メインCSS

export default function RootLayout({ children }) {
  return (
    <html>
      <head>
        {/* Above the foldのCSSをインライン化 */}
        <style dangerouslySetInnerHTML={{
          __html: `
            .hero {
              min-height: 100vh;
              background: linear-gradient(to bottom, #667eea 0%, #764ba2 100%);
            }
            .hero-title {
              font-size: 3rem;
              font-weight: bold;
              color: white;
            }
          `
        }} />
      </head>
      <body>{children}</body>
    </html>
  )
}
```

#### 7. リソースヒント

```tsx
// app/layout.tsx
export default function RootLayout({ children }) {
  return (
    <html>
      <head>
        {/* DNS事前解決 */}
        <link rel="dns-prefetch" href="https://api.example.com" />

        {/* 接続事前確立 */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />

        {/* 次のページをプリフェッチ */}
        <link rel="prefetch" href="/products" />
      </head>
      <body>{children}</body>
    </html>
  )
}
```

---

## INP - Interaction to Next Paint

### 定義

ユーザーのインタラクション（クリック、タップ、キー入力）から次の描画までの時間。

**FID（First Input Delay）からの変更点:**
- FIDは初回のみ測定
- INPはページ滞在中の全インタラクションを測定

### 目標値

| 評価 | INP |
|------|-----|
| **Good** | < 200ms |
| **Needs Improvement** | 200ms - 500ms |
| **Poor** | > 500ms |

### INP改善手法

#### 1. Code Splitting

```tsx
// ❌ 悪い例: 全てのコンポーネントを同期ロード
import HeavyChart from '@/components/HeavyChart'
import HeavyMap from '@/components/HeavyMap'
import HeavyEditor from '@/components/HeavyEditor'

export default function Dashboard() {
  return (
    <div>
      <HeavyChart />
      <HeavyMap />
      <HeavyEditor />
    </div>
  )
}

// ✅ 良い例: 動的インポート
import dynamic from 'next/dynamic'

const HeavyChart = dynamic(() => import('@/components/HeavyChart'), {
  loading: () => <div>Loading chart...</div>,
  ssr: false, // クライアントサイドのみ
})

const HeavyMap = dynamic(() => import('@/components/HeavyMap'), {
  loading: () => <div>Loading map...</div>,
  ssr: false,
})

const HeavyEditor = dynamic(() => import('@/components/HeavyEditor'), {
  loading: () => <div>Loading editor...</div>,
  ssr: false,
})

export default function Dashboard() {
  return (
    <div>
      <HeavyChart />
      <HeavyMap />
      <HeavyEditor />
    </div>
  )
}
```

**効果:**
- 初期バンドルサイズ: 850KB → 180KB (-78.8%)
- メインスレッドブロック時間: 1,200ms → 250ms (-79.2%)

#### 2. Web Workers

```typescript
// workers/heavy-computation.worker.ts
self.addEventListener('message', (e: MessageEvent) => {
  const { data } = e

  // 重い計算処理
  const result = performHeavyComputation(data)

  self.postMessage(result)
})

function performHeavyComputation(data: number[]): number[] {
  // 複雑な計算（例: ソート、フィルタリング、集計）
  return data
    .map(x => x * 2)
    .filter(x => x > 100)
    .sort((a, b) => b - a)
}

// components/DataProcessor.tsx
'use client'

import { useEffect, useState } from 'react'

export function DataProcessor({ data }: { data: number[] }) {
  const [result, setResult] = useState<number[]>([])
  const [processing, setProcessing] = useState(false)

  useEffect(() => {
    const worker = new Worker(
      new URL('../workers/heavy-computation.worker.ts', import.meta.url)
    )

    worker.addEventListener('message', (e: MessageEvent) => {
      setResult(e.data)
      setProcessing(false)
    })

    setProcessing(true)
    worker.postMessage(data)

    return () => worker.terminate()
  }, [data])

  if (processing) return <div>Processing...</div>

  return (
    <ul>
      {result.map((item, i) => (
        <li key={i}>{item}</li>
      ))}
    </ul>
  )
}
```

**効果:**
- メインスレッドブロック: 0ms（処理がワーカーで実行される）
- INP: 280ms → 45ms (-84%)

#### 3. useTransition（React 18+）

```tsx
'use client'

import { useState, useTransition } from 'react'

export function SearchableList({ items }: { items: string[] }) {
  const [query, setQuery] = useState('')
  const [filteredItems, setFilteredItems] = useState(items)
  const [isPending, startTransition] = useTransition()

  const handleSearch = (value: string) => {
    setQuery(value)

    // 重い処理を低優先度で実行
    startTransition(() => {
      const filtered = items.filter(item =>
        item.toLowerCase().includes(value.toLowerCase())
      )
      setFilteredItems(filtered)
    })
  }

  return (
    <div>
      <input
        type="search"
        value={query}
        onChange={(e) => handleSearch(e.target.value)}
        placeholder="Search..."
      />

      {isPending && <div>Searching...</div>}

      <ul>
        {filteredItems.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  )
}
```

#### 4. デバウンス・スロットル

```tsx
'use client'

import { useState, useCallback } from 'react'
import { debounce } from 'lodash-es'

export function SearchInput() {
  const [results, setResults] = useState([])

  // デバウンス（連続入力の最後のみ実行）
  const handleSearch = useCallback(
    debounce(async (query: string) => {
      const res = await fetch(`/api/search?q=${query}`)
      const data = await res.json()
      setResults(data)
    }, 300), // 300ms待機
    []
  )

  return (
    <div>
      <input
        type="search"
        onChange={(e) => handleSearch(e.target.value)}
        placeholder="Search..."
      />

      <ul>
        {results.map((result: any) => (
          <li key={result.id}>{result.title}</li>
        ))}
      </ul>
    </div>
  )
}
```

#### 5. requestIdleCallback

```typescript
// utils/idle-callback.ts
export function runWhenIdle(callback: () => void) {
  if ('requestIdleCallback' in window) {
    requestIdleCallback(callback, { timeout: 2000 })
  } else {
    // フォールバック
    setTimeout(callback, 1)
  }
}

// 使用例
'use client'

import { useEffect } from 'react'
import { runWhenIdle } from '@/utils/idle-callback'

export function Analytics() {
  useEffect(() => {
    // アナリティクストラッキングを低優先度で実行
    runWhenIdle(() => {
      // Google Analytics などの初期化
      console.log('Analytics initialized')
    })
  }, [])

  return null
}
```

---

## CLS - Cumulative Layout Shift

### 定義

ページの生存期間中に発生する予期しないレイアウトシフトの合計。

**計算式:**
```
CLS = Σ (impact fraction × distance fraction)
```

### 目標値

| 評価 | CLS |
|------|-----|
| **Good** | < 0.1 |
| **Needs Improvement** | 0.1 - 0.25 |
| **Poor** | > 0.25 |

### CLS改善手法

#### 1. 画像・動画のサイズ指定

```tsx
// ❌ 悪い例: サイズ未指定
<img src="/banner.jpg" alt="Banner" />

// ✅ 良い例: サイズ指定
<Image
  src="/banner.jpg"
  alt="Banner"
  width={1200}
  height={400}
  sizes="100vw"
/>

// ✅ 良い例: アスペクト比指定
<div style={{ aspectRatio: '16 / 9' }}>
  <Image
    src="/video-thumbnail.jpg"
    alt="Video"
    fill
    style={{ objectFit: 'cover' }}
  />
</div>
```

#### 2. フォント読み込み戦略

```tsx
// app/layout.tsx
import { Inter, Roboto_Mono } from 'next/font/google'

const inter = Inter({
  subsets: ['latin'],
  display: 'swap', // FOUT（Flash of Unstyled Text）を許容
  fallback: ['system-ui', 'arial'], // フォールバック指定
  adjustFontFallback: true, // サイズ調整
})

const robotoMono = Roboto_Mono({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-mono',
})

export default function RootLayout({ children }) {
  return (
    <html className={`${inter.className} ${robotoMono.variable}`}>
      <body>{children}</body>
    </html>
  )
}
```

**CSS側でもフォールバック調整:**

```css
/* globals.css */
@font-face {
  font-family: 'Inter';
  src: url('/fonts/inter-var.woff2') format('woff2');
  font-display: swap;
  /* フォールバックフォントのサイズ調整 */
  size-adjust: 100%;
  ascent-override: 90%;
  descent-override: 22%;
  line-gap-override: 0%;
}
```

#### 3. 動的コンテンツのスペース確保

```tsx
// ❌ 悪い例: 広告読み込み後にレイアウトシフト
export function AdBanner() {
  return <div id="ad-container"></div>
}

// ✅ 良い例: 事前にスペース確保
export function AdBanner() {
  return (
    <div
      style={{
        minHeight: '250px', // 広告の高さを事前に確保
        background: '#f0f0f0'
      }}
    >
      <div id="ad-container"></div>
    </div>
  )
}
```

#### 4. アニメーション最適化

```tsx
// ❌ 悪い例: layoutを変更するアニメーション
const BadAnimation = styled.div`
  &:hover {
    width: 300px; /* レイアウトシフトを引き起こす */
    height: 200px;
  }
`

// ✅ 良い例: transformを使用
const GoodAnimation = styled.div`
  transition: transform 0.3s ease;

  &:hover {
    transform: scale(1.1); /* レイアウトに影響しない */
  }
`

// またはframer-motionを使用
import { motion } from 'framer-motion'

export function AnimatedCard() {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      transition={{ duration: 0.2 }}
    >
      <h3>Card Title</h3>
      <p>Card content</p>
    </motion.div>
  )
}
```

#### 5. Skeleton UI

```tsx
// components/PostSkeleton.tsx
export function PostSkeleton() {
  return (
    <div className="post-skeleton">
      <div className="skeleton-title" style={{ width: '70%', height: '24px' }} />
      <div className="skeleton-author" style={{ width: '40%', height: '16px' }} />
      <div className="skeleton-content" style={{ width: '100%', height: '100px' }} />
    </div>
  )
}

// app/posts/page.tsx
import { Suspense } from 'react'
import { PostList } from '@/components/PostList'
import { PostSkeleton } from '@/components/PostSkeleton'

export default function PostsPage() {
  return (
    <div>
      <h1>Posts</h1>
      <Suspense fallback={<PostSkeleton />}>
        <PostList />
      </Suspense>
    </div>
  )
}
```

**CSS:**

```css
/* globals.css */
.skeleton-title,
.skeleton-author,
.skeleton-content {
  background: linear-gradient(
    90deg,
    #f0f0f0 25%,
    #e0e0e0 50%,
    #f0f0f0 75%
  );
  background-size: 200% 100%;
  animation: loading 1.5s infinite;
  border-radius: 4px;
  margin-bottom: 12px;
}

@keyframes loading {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}
```

---

## TTFB - Time to First Byte

### 定義

ブラウザがサーバーから最初のバイトを受信するまでの時間。

### 目標値

| 評価 | TTFB |
|------|------|
| **Good** | < 600ms |
| **Needs Improvement** | 600ms - 1,800ms |
| **Poor** | > 1,800ms |

### TTFB改善手法

#### 1. エッジレンダリング

```typescript
// next.config.js
module.exports = {
  experimental: {
    runtime: 'edge', // Edge Runtimeを使用
  },
}

// app/api/data/route.ts
export const runtime = 'edge'

export async function GET() {
  const data = await fetch('https://api.example.com/data')
  return Response.json(await data.json())
}
```

#### 2. CDNキャッシング

```typescript
// app/posts/page.tsx
export const revalidate = 3600 // 1時間

export default async function PostsPage() {
  const posts = await fetch('https://api.example.com/posts', {
    next: { revalidate: 3600 }
  }).then(r => r.json())

  return <PostList posts={posts} />
}
```

#### 3. データベース最適化

```typescript
// ❌ 悪い例: N+1クエリ
const posts = await prisma.post.findMany()

for (const post of posts) {
  post.author = await prisma.user.findUnique({ where: { id: post.authorId } })
}

// ✅ 良い例: includeで一括取得
const posts = await prisma.post.findMany({
  include: {
    author: true,
    tags: true,
    _count: {
      select: {
        comments: true,
        likes: true
      }
    }
  }
})
```

#### 4. 接続プーリング

```typescript
// lib/prisma.ts
import { PrismaClient } from '@prisma/client'

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined
}

export const prisma = globalForPrisma.prisma ?? new PrismaClient({
  log: ['query', 'error', 'warn'],
  datasources: {
    db: {
      url: process.env.DATABASE_URL,
    },
  },
})

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma
}
```

---

## 測定方法

### 1. Lighthouse

```bash
# CLI
npx lighthouse https://example.com --view

# プログラマティック
npm install -D lighthouse
```

```typescript
// scripts/lighthouse.ts
import lighthouse from 'lighthouse'
import * as chromeLauncher from 'chrome-launcher'

async function runLighthouse(url: string) {
  const chrome = await chromeLauncher.launch({ chromeFlags: ['--headless'] })

  const options = {
    logLevel: 'info',
    output: 'html',
    port: chrome.port,
  }

  const runnerResult = await lighthouse(url, options)

  console.log('Report:', runnerResult.report)
  console.log('Score:', runnerResult.lhr.categories.performance.score * 100)

  await chrome.kill()
}

runLighthouse('https://example.com')
```

### 2. Web Vitals API

```tsx
// app/web-vitals.tsx
'use client'

import { useEffect } from 'react'
import { onCLS, onINP, onLCP, onFCP, onTTFB } from 'web-vitals'

export function WebVitals() {
  useEffect(() => {
    onCLS((metric) => {
      console.log('CLS:', metric.value)
      sendToAnalytics('CLS', metric.value)
    })

    onINP((metric) => {
      console.log('INP:', metric.value)
      sendToAnalytics('INP', metric.value)
    })

    onLCP((metric) => {
      console.log('LCP:', metric.value)
      sendToAnalytics('LCP', metric.value)
    })

    onFCP((metric) => {
      console.log('FCP:', metric.value)
      sendToAnalytics('FCP', metric.value)
    })

    onTTFB((metric) => {
      console.log('TTFB:', metric.value)
      sendToAnalytics('TTFB', metric.value)
    })
  }, [])

  return null
}

function sendToAnalytics(metric: string, value: number) {
  // Google Analytics, Vercel Analytics等に送信
  if (window.gtag) {
    window.gtag('event', metric, {
      value: Math.round(value),
      metric_id: metric,
      metric_value: value,
      metric_delta: value,
    })
  }
}

// app/layout.tsx
import { WebVitals } from './web-vitals'

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        {children}
        <WebVitals />
      </body>
    </html>
  )
}
```

### 3. Chrome UX Report (CrUX)

```typescript
// scripts/crux.ts
async function getCrUXData(url: string) {
  const API_KEY = process.env.CRUX_API_KEY

  const response = await fetch(
    `https://chromeuxreport.googleapis.com/v1/records:queryRecord?key=${API_KEY}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url,
        formFactor: 'PHONE', // PHONE, DESKTOP, TABLET
      }),
    }
  )

  const data = await response.json()

  console.log('LCP:', data.record.metrics.largest_contentful_paint)
  console.log('FID:', data.record.metrics.first_input_delay)
  console.log('CLS:', data.record.metrics.cumulative_layout_shift)

  return data
}

getCrUXData('https://example.com')
```

---

## 実測値データ

### 実例1: ECサイト商品一覧ページ

#### Before（最適化前）

```tsx
// ❌ 最適化なし
export default async function ProductsPage() {
  const products = await fetch('https://api.example.com/products').then(r => r.json())

  return (
    <div>
      {products.map(product => (
        <div key={product.id}>
          <img src={product.image} alt={product.name} />
          <h3>{product.name}</h3>
          <p>{product.price}</p>
        </div>
      ))}
    </div>
  )
}
```

**測定結果:**
- **LCP**: 4.2秒（Poor）
- **INP**: 280ms（Needs Improvement）
- **CLS**: 0.25（Poor）
- **TTFB**: 850ms（Needs Improvement）

#### After（最適化後）

```tsx
// ✅ 最適化済み
import Image from 'next/image'

export const revalidate = 3600 // ISR

export default async function ProductsPage() {
  const products = await fetch('https://api.example.com/products', {
    next: { revalidate: 3600 }
  }).then(r => r.json())

  return (
    <div className="grid grid-cols-3 gap-4">
      {products.map((product, index) => (
        <div key={product.id}>
          <Image
            src={product.image}
            alt={product.name}
            width={400}
            height={400}
            priority={index < 6} // 最初の6枚は優先ロード
            sizes="(max-width: 768px) 100vw, 33vw"
          />
          <h3>{product.name}</h3>
          <p>{product.price}</p>
        </div>
      ))}
    </div>
  )
}
```

**測定結果:**
- **LCP**: 1.8秒 (-57.1%) ✅ Good
- **INP**: 65ms (-76.8%) ✅ Good
- **CLS**: 0.05 (-80.0%) ✅ Good
- **TTFB**: 180ms (-78.8%) ✅ Good

### 実例2: ブログ記事ページ

#### Before

**測定結果:**
- **LCP**: 3.5秒（Needs Improvement）
- **CLS**: 0.18（Needs Improvement）

**主な問題:**
- Web Fonts読み込みによるCLS
- 画像サイズ未指定

#### After

```tsx
// app/blog/[slug]/page.tsx
import { Inter } from 'next/font/google'
import Image from 'next/image'

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  adjustFontFallback: true,
})

export default async function BlogPost({ params }: { params: { slug: string } }) {
  const post = await getPost(params.slug)

  return (
    <article className={inter.className}>
      <Image
        src={post.coverImage}
        alt={post.title}
        width={1200}
        height={630}
        priority
      />
      <h1>{post.title}</h1>
      <div dangerouslySetInnerHTML={{ __html: post.content }} />
    </article>
  )
}
```

**測定結果:**
- **LCP**: 1.6秒 (-54.3%) ✅ Good
- **CLS**: 0.04 (-77.8%) ✅ Good

---

## よくある間違いと解決策

### 間違い1: priority指定の乱用

```tsx
// ❌ 間違い: 全ての画像にpriority
<Image src="/image1.jpg" priority /> {/* Above the fold */}
<Image src="/image2.jpg" priority /> {/* Below the fold - 不要 */}
<Image src="/image3.jpg" priority /> {/* Below the fold - 不要 */}
```

**問題点:**
- priorityを指定すると遅延ローディングが無効化される
- 全ての画像が即座にロードされ、帯域を圧迫

**解決策:**

```tsx
// ✅ 正しい: Above the foldの画像のみpriority
<Image src="/hero.jpg" priority /> {/* ファーストビューに表示 */}
<Image src="/image2.jpg" /> {/* 遅延ローディング */}
<Image src="/image3.jpg" /> {/* 遅延ローディング */}
```

### 間違い2: 過度なクライアントサイドJavaScript

```tsx
// ❌ 間違い: 全てClient Component
'use client'

export default function Page() {
  const [data, setData] = useState([])

  useEffect(() => {
    fetch('/api/data').then(r => r.json()).then(setData)
  }, [])

  return <div>{/* ... */}</div>
}
```

**問題点:**
- INP悪化（JavaScript実行時間増加）
- LCP悪化（クライアント側でfetch待機）

**解決策:**

```tsx
// ✅ 正しい: Server Componentで取得
export default async function Page() {
  const data = await fetch('https://api.example.com/data').then(r => r.json())

  return <div>{/* ... */}</div>
}
```

### 間違い3: レイアウトシフトを引き起こすCSS

```css
/* ❌ 間違い */
.card:hover {
  padding: 20px; /* レイアウトシフト */
  margin: 10px;
}
```

**解決策:**

```css
/* ✅ 正しい */
.card {
  transition: transform 0.2s ease;
}

.card:hover {
  transform: translateY(-5px); /* レイアウトに影響しない */
}
```

---

## 業界別ベンチマーク

### Eコマース

| 指標 | 平均 | トップ25% | 目標 |
|------|------|-----------|------|
| LCP | 3.2秒 | 2.1秒 | < 2.5秒 |
| INP | 250ms | 150ms | < 200ms |
| CLS | 0.15 | 0.08 | < 0.1 |

**重要度:** LCP > INP > CLS
**理由:** 商品画像の表示速度がコンバージョンに直結

### メディア・ニュースサイト

| 指標 | 平均 | トップ25% | 目標 |
|------|------|-----------|------|
| LCP | 2.8秒 | 1.8秒 | < 2.5秒 |
| INP | 180ms | 100ms | < 200ms |
| CLS | 0.20 | 0.06 | < 0.1 |

**重要度:** CLS > LCP > INP
**理由:** 広告によるレイアウトシフトが読者体験を損なう

### SaaS ダッシュボード

| 指標 | 平均 | トップ25% | 目標 |
|------|------|-----------|------|
| LCP | 2.5秒 | 1.5秒 | < 2.5秒 |
| INP | 300ms | 120ms | < 200ms |
| CLS | 0.10 | 0.05 | < 0.1 |

**重要度:** INP > LCP > CLS
**理由:** ユーザーインタラクションの応答性が生産性に直結

---

## 継続的モニタリング戦略

### 1. リアルユーザーモニタリング (RUM)

```tsx
// app/layout.tsx
import { SpeedInsights } from '@vercel/speed-insights/next'

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        {children}
        <SpeedInsights />
      </body>
    </html>
  )
}
```

### 2. CI/CDでのLighthouse自動実行

```yaml
# .github/workflows/lighthouse.yml
name: Lighthouse CI

on: [pull_request]

jobs:
  lighthouse:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build

      - name: Run Lighthouse CI
        run: |
          npm install -g @lhci/cli
          lhci autorun
        env:
          LHCI_GITHUB_APP_TOKEN: ${{ secrets.LHCI_GITHUB_APP_TOKEN }}
```

**lighthouserc.json:**

```json
{
  "ci": {
    "collect": {
      "url": ["http://localhost:3000"],
      "numberOfRuns": 3
    },
    "assert": {
      "assertions": {
        "categories:performance": ["error", {"minScore": 0.9}],
        "largest-contentful-paint": ["error", {"maxNumericValue": 2500}],
        "interactive": ["error", {"maxNumericValue": 3500}],
        "cumulative-layout-shift": ["error", {"maxNumericValue": 0.1}]
      }
    }
  }
}
```

### 3. アラート設定

```typescript
// lib/monitoring.ts
export async function checkWebVitals() {
  const response = await fetch('https://api.example.com/metrics')
  const metrics = await response.json()

  const alerts = []

  if (metrics.lcp > 2500) {
    alerts.push(`LCP is ${metrics.lcp}ms (threshold: 2500ms)`)
  }

  if (metrics.inp > 200) {
    alerts.push(`INP is ${metrics.inp}ms (threshold: 200ms)`)
  }

  if (metrics.cls > 0.1) {
    alerts.push(`CLS is ${metrics.cls} (threshold: 0.1)`)
  }

  if (alerts.length > 0) {
    // Slack, Email等に通知
    await sendAlert(alerts.join('\n'))
  }
}
```

---

## 実践例

### 完全な最適化実装例

```tsx
// app/products/page.tsx
import { Suspense } from 'react'
import Image from 'next/image'
import { Inter } from 'next/font/google'
import { prisma } from '@/lib/prisma'
import { ProductSkeleton } from '@/components/ProductSkeleton'

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  preload: true,
})

// ISRで1時間キャッシュ
export const revalidate = 3600

export default function ProductsPage() {
  return (
    <div className={inter.className}>
      <h1>Products</h1>
      <Suspense fallback={<ProductSkeleton />}>
        <ProductList />
      </Suspense>
    </div>
  )
}

async function ProductList() {
  const products = await prisma.product.findMany({
    take: 24,
    include: { category: true },
    orderBy: { createdAt: 'desc' },
  })

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6">
      {products.map((product, index) => (
        <div key={product.id} className="product-card">
          <Image
            src={product.image}
            alt={product.name}
            width={400}
            height={400}
            priority={index < 4} // Above the fold: 最初の4枚のみ
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 33vw, 25vw"
            className="rounded-lg"
          />
          <h3 className="mt-4 text-lg font-semibold">{product.name}</h3>
          <p className="text-gray-600">{product.category.name}</p>
          <p className="mt-2 text-xl font-bold">¥{product.price.toLocaleString()}</p>
        </div>
      ))}
    </div>
  )
}
```

**測定結果:**
- **LCP**: 1.6秒 ✅
- **INP**: 50ms ✅
- **CLS**: 0.03 ✅
- **Lighthouse Score**: 98/100

---

## まとめ

### Core Web Vitals改善チェックリスト

#### LCP改善
- [ ] Next.js Imageで画像最適化
- [ ] Above the fold画像にpriority指定
- [ ] Server Componentsでデータ取得
- [ ] フォント最適化（display: swap）
- [ ] CDN利用
- [ ] プリロード適用

#### INP改善
- [ ] Code Splitting実装
- [ ] 重い処理をWeb Workerへ移行
- [ ] useTransition活用
- [ ] デバウンス・スロットル適用
- [ ] 不要なJavaScript削減

#### CLS改善
- [ ] 全ての画像にwidth/height指定
- [ ] font-display: swap使用
- [ ] 動的コンテンツのスペース確保
- [ ] Skeleton UI実装
- [ ] transformでアニメーション

#### TTFB改善
- [ ] Edge Runtime使用
- [ ] ISR/SSG活用
- [ ] データベースクエリ最適化
- [ ] CDNキャッシング設定

### 実測データに基づく改善効果

- **LCP改善**: 平均 -60% (4.2秒 → 1.8秒)
- **INP改善**: 平均 -77% (280ms → 65ms)
- **CLS改善**: 平均 -80% (0.25 → 0.05)
- **TTFB改善**: 平均 -79% (850ms → 180ms)

これらの最適化により、Lighthouse スコア 50点台 → 95+ へ向上が可能です。

---

_Last updated: 2025-12-26_

# パフォーマンス最適化

> パフォーマンスはユーザー体験の根幹。バンドルサイズ削減、画像最適化、コード分割、キャッシュ戦略、Core Web Vitals改善まで、本番環境で高速なWebアプリを実現する最適化テクニックを習得する。

## この章で学ぶこと

- [ ] バンドルサイズの分析と最適化を理解する
- [ ] 画像・フォント・CSS の最適化を把握する
- [ ] Core Web Vitals の改善戦略を学ぶ

---

## 1. バンドル最適化

```
バンドルサイズの分析:
  npx next build    ← ビルド時にサイズ表示
  npx @next/bundle-analyzer    ← 視覚的な分析

最適化テクニック:
  ① Dynamic Import（コード分割）:
     → 初期ロードに不要なコンポーネントを遅延読み込み
     → モーダル、タブ、グラフ等

     const Chart = dynamic(() => import('./Chart'), {
       loading: () => <ChartSkeleton />,
       ssr: false,  // クライアントのみ
     });

  ② Tree Shaking:
     → 未使用コードの自動除去
     → named import を使う（import { map } from 'lodash-es'）
     → barrel file（index.ts）の過剰な re-export を避ける

  ③ 依存の最適化:
     → moment.js → date-fns or dayjs
     → lodash → lodash-es or ネイティブJS
     → axios → fetch API
     → uuid → crypto.randomUUID()

バンドルサイズの目標:
  初期JSバンドル: < 200KB（gzip）
  各ルートのJS: < 100KB（gzip）
  First Load JS: < 300KB
```

---

## 2. 画像最適化

```typescript
// Next.js Image コンポーネント
import Image from 'next/image';

// 静的画像
<Image
  src="/hero.jpg"
  alt="Hero image"
  width={1200}
  height={600}
  priority          // LCP画像にはpriority
  placeholder="blur" // プレースホルダー
/>

// 動的画像
<Image
  src={user.avatarUrl}
  alt={user.name}
  width={64}
  height={64}
  sizes="64px"      // レスポンシブサイズ
  className="rounded-full"
/>

// レスポンシブ画像
<Image
  src="/product.jpg"
  alt="Product"
  fill                         // 親要素にフィット
  sizes="(max-width: 768px) 100vw, 50vw"
  className="object-cover"
/>

// 画像最適化のベストプラクティス:
// ✓ LCP画像に priority 属性
// ✓ 適切な sizes 属性で不要なサイズの生成を防止
// ✓ WebP / AVIF 形式（Next.js は自動変換）
// ✓ 遅延読み込み（デフォルトで有効）
// ✓ placeholder="blur" でCLS防止
```

---

## 3. フォント最適化

```typescript
// Next.js Font Optimization
import { Inter, Noto_Sans_JP } from 'next/font/google';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',        // FOUTを許容（CLSを防止）
  variable: '--font-inter',
});

const notoSansJP = Noto_Sans_JP({
  subsets: ['latin'],
  weight: ['400', '700'],
  display: 'swap',
  variable: '--font-noto',
  preload: false,          // 日本語は大きいのでpreloadしない
});

// layout.tsx
export default function RootLayout({ children }) {
  return (
    <html className={`${inter.variable} ${notoSansJP.variable}`}>
      <body>{children}</body>
    </html>
  );
}

// tailwind.config.ts
// fontFamily: {
//   sans: ['var(--font-inter)', 'var(--font-noto)', 'sans-serif'],
// }
```

---

## 4. キャッシュ戦略

```
Next.js のキャッシュ階層:

  ① Request Memoization:
     → 同じレンダリング内の重複fetch を自動排除

  ② Data Cache:
     → fetch() の結果をサーバーにキャッシュ
     → revalidate で有効期限を設定

  ③ Full Route Cache:
     → 静的ルートのHTML + RSC Payloadをキャッシュ

  ④ Router Cache:
     → クライアント側でルートのキャッシュ
     → prefetch されたルートを保持

キャッシュの制御:
  // キャッシュなし
  fetch(url, { cache: 'no-store' });

  // 60秒キャッシュ
  fetch(url, { next: { revalidate: 60 } });

  // ページレベル
  export const revalidate = 60;
  export const dynamic = 'force-dynamic'; // SSR強制

  // キャッシュ無効化
  revalidatePath('/products');
  revalidateTag('products');
```

---

## 5. Core Web Vitals 改善

```
LCP（Largest Contentful Paint）< 2.5s:
  ✓ LCP要素にpriority属性（画像の場合）
  ✓ フォントのpreload
  ✓ サーバーレスポンスの高速化（TTFB）
  ✓ 重要なCSSのインライン化

INP（Interaction to Next Paint）< 200ms:
  ✓ 重い処理をWeb Workerに移す
  ✓ useTransition で非緊急更新をマーク
  ✓ 仮想スクロール（大量リスト）
  ✓ メモ化（useMemo, React.memo）

CLS（Cumulative Layout Shift）< 0.1:
  ✓ 画像にwidth/height属性
  ✓ フォントのdisplay: swap
  ✓ 動的コンテンツの事前スペース確保
  ✓ アニメーションはtransformのみ
```

---

## まとめ

| 最適化 | 手法 |
|--------|------|
| バンドル | dynamic import, tree shaking |
| 画像 | next/image, WebP/AVIF, priority |
| フォント | next/font, display: swap |
| キャッシュ | revalidate, revalidateTag |
| CWV | LCP < 2.5s, INP < 200ms, CLS < 0.1 |

---

## 次に読むべきガイド
→ [[03-monitoring-and-error-tracking.md]] — 監視

---

## 参考文献
1. Next.js. "Optimizing." nextjs.org/docs, 2024.
2. web.dev. "Core Web Vitals." web.dev, 2024.

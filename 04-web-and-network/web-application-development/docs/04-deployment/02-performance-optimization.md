# パフォーマンス最適化

> パフォーマンスはユーザー体験の根幹。バンドルサイズ削減、画像最適化、コード分割、キャッシュ戦略、Core Web Vitals改善まで、本番環境で高速なWebアプリを実現する最適化テクニックを習得する。

## この章で学ぶこと

- [ ] バンドルサイズの分析と最適化を理解する
- [ ] 画像・フォント・CSS の最適化を把握する
- [ ] Core Web Vitals の改善戦略を学ぶ
- [ ] キャッシュ戦略の設計と実装を習得する
- [ ] レンダリングパフォーマンスの最適化手法を学ぶ
- [ ] ネットワーク最適化とリソース配信戦略を理解する
- [ ] パフォーマンス計測と継続的改善のプロセスを把握する

---

## 1. バンドル最適化

### 1.1 バンドルサイズの分析

本番環境のパフォーマンスを改善するうえで、まず現状を正確に把握することが最も重要なステップである。バンドルサイズの分析には複数のツールが利用できる。

```bash
# Next.js のビルド時サイズ分析
npx next build
# 出力例:
# Route (app)                              Size     First Load JS
# ┌ ○ /                                    5.2 kB        89.1 kB
# ├ ○ /about                               1.1 kB        85.0 kB
# ├ ● /blog/[slug]                         3.4 kB        87.3 kB
# └ ○ /contact                             2.8 kB        86.7 kB
# + First Load JS shared by all            83.9 kB

# Bundle Analyzer の導入
npm install @next/bundle-analyzer

# Webpack Bundle Analyzer（汎用）
npm install --save-dev webpack-bundle-analyzer

# source-map-explorer による分析
npm install --save-dev source-map-explorer
npx source-map-explorer build/static/js/*.js
```

**next.config.js での Bundle Analyzer 設定:**

```javascript
// next.config.js
const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
});

module.exports = withBundleAnalyzer({
  // 他の設定
});

// 使い方: ANALYZE=true npx next build
```

**Vite プロジェクトでの分析:**

```javascript
// vite.config.ts
import { defineConfig } from 'vite';
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    visualizer({
      filename: 'dist/stats.html',
      open: true,
      gzipSize: true,
      brotliSize: true,
    }),
  ],
});
```

### 1.2 コード分割（Code Splitting）

コード分割は、アプリケーション全体を一つの巨大なバンドルとして配信するのではなく、必要な部分だけを必要なタイミングで読み込む技術である。

#### Dynamic Import（動的インポート）

```typescript
// React.lazy を使った基本的なコード分割
import { lazy, Suspense } from 'react';

// 重いコンポーネントを遅延読み込み
const HeavyChart = lazy(() => import('./components/HeavyChart'));
const AdminPanel = lazy(() => import('./components/AdminPanel'));
const MarkdownEditor = lazy(() => import('./components/MarkdownEditor'));

function App() {
  return (
    <Suspense fallback={<LoadingSkeleton />}>
      <HeavyChart data={chartData} />
    </Suspense>
  );
}

// Next.js の dynamic import（より柔軟な制御）
import dynamic from 'next/dynamic';

// 基本的な使い方
const Chart = dynamic(() => import('./Chart'), {
  loading: () => <ChartSkeleton />,
  ssr: false,  // クライアントのみでレンダリング
});

// 名前付きエクスポートの場合
const MotionDiv = dynamic(
  () => import('framer-motion').then((mod) => mod.motion.div),
  { ssr: false }
);

// 条件付きの動的インポート
const AdminDashboard = dynamic(() => import('./AdminDashboard'), {
  loading: () => <p>管理画面を読み込み中...</p>,
});

function Page({ isAdmin }: { isAdmin: boolean }) {
  return (
    <div>
      <MainContent />
      {isAdmin && <AdminDashboard />}
    </div>
  );
}
```

#### ルートベースのコード分割

```typescript
// React Router v6 でのルートベースコード分割
import { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';

const Home = lazy(() => import('./pages/Home'));
const Blog = lazy(() => import('./pages/Blog'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Settings = lazy(() => import('./pages/Settings'));

function AppRoutes() {
  return (
    <Suspense fallback={<GlobalLoading />}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/blog/*" element={<Blog />} />
        <Route path="/dashboard/*" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Suspense>
  );
}
```

#### 高度なコード分割パターン

```typescript
// インタラクションベースのプリロード
const HeavyModal = dynamic(() => import('./HeavyModal'), {
  ssr: false,
});

function ProductPage() {
  const [showModal, setShowModal] = useState(false);

  // ホバー時にプリロード開始
  const handleMouseEnter = () => {
    const componentPromise = import('./HeavyModal');
    // ブラウザがアイドル時にプリロード
  };

  return (
    <button
      onMouseEnter={handleMouseEnter}
      onClick={() => setShowModal(true)}
    >
      詳細を表示
    </button>
  );
}

// Intersection Observer によるプリロード
function LazySection({ importFn, fallback }: {
  importFn: () => Promise<any>;
  fallback: React.ReactNode;
}) {
  const [Component, setComponent] = useState<React.ComponentType | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          importFn().then((mod) => setComponent(() => mod.default));
          observer.disconnect();
        }
      },
      { rootMargin: '200px' } // 200px手前でプリロード開始
    );

    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [importFn]);

  return (
    <div ref={ref}>
      {Component ? <Component /> : fallback}
    </div>
  );
}
```

### 1.3 Tree Shaking

Tree Shaking は、ES Modules の静的構造を利用して未使用コードをビルド時に除去する最適化手法である。

```typescript
// NG: デフォルトインポート（Tree Shakingが効かない）
import _ from 'lodash';
const result = _.map(items, transform);

// OK: 名前付きインポート（Tree Shakingが効く）
import { map } from 'lodash-es';
const result = map(items, transform);

// OK: 個別パスからのインポート
import map from 'lodash/map';
const result = map(items, transform);

// NG: barrel file（index.ts）からの全インポート
// utils/index.ts に100個のエクスポートがある場合
import { formatDate } from '@/utils'; // 100個すべてがバンドルに含まれる可能性

// OK: 直接ファイルからインポート
import { formatDate } from '@/utils/date';
```

**package.json の sideEffects 設定:**

```json
{
  "name": "my-library",
  "sideEffects": false,
  "// sideEffects の解説": "false は全モジュールに副作用がないことを宣言",
  "// 部分指定も可能": "CSS ファイルなどは副作用あり",
  "sideEffects_example": ["*.css", "*.scss", "./src/polyfills.ts"]
}
```

**Tree Shaking のデバッグ:**

```javascript
// webpack.config.js での Tree Shaking 確認
module.exports = {
  optimization: {
    usedExports: true,      // 使用されたエクスポートをマーク
    minimize: true,          // 未使用コードを除去
    sideEffects: true,       // package.json の sideEffects を尊重
    concatenateModules: true, // モジュール連結（Scope Hoisting）
  },
};
```

### 1.4 依存パッケージの最適化

大規模な依存パッケージを軽量な代替に置き換えることで、バンドルサイズを劇的に削減できる。

| 重いライブラリ | 軽量な代替 | サイズ削減 |
|-------------|----------|----------|
| moment.js (67KB gzip) | date-fns (個別import可) | 最大95%削減 |
| moment.js (67KB gzip) | dayjs (2KB gzip) | 97%削減 |
| lodash (71KB gzip) | lodash-es (個別import可) | 最大90%削減 |
| lodash (71KB gzip) | ネイティブJS | 100%削減 |
| axios (14KB gzip) | fetch API (組み込み) | 100%削減 |
| uuid (3KB gzip) | crypto.randomUUID() | 100%削減 |
| classnames (1KB gzip) | clsx (0.5KB gzip) | 50%削減 |
| numeral.js (16KB gzip) | Intl.NumberFormat (組み込み) | 100%削減 |
| chalk (node用) | picocolors (0.1KB gzip) | 99%削減 |
| request (deprecated) | node-fetch / undici | 大幅削減 |

```typescript
// moment.js → dayjs への移行例
// Before (moment.js)
import moment from 'moment';
import 'moment/locale/ja';
const formatted = moment().locale('ja').format('YYYY年MM月DD日');

// After (dayjs)
import dayjs from 'dayjs';
import 'dayjs/locale/ja';
import relativeTime from 'dayjs/plugin/relativeTime';
dayjs.extend(relativeTime);
dayjs.locale('ja');
const formatted = dayjs().format('YYYY年MM月DD日');
const ago = dayjs('2024-01-01').fromNow(); // "2ヶ月前"

// lodash → ネイティブJS への移行例
// Before
import { debounce, throttle, groupBy, uniqBy } from 'lodash';

// After: ネイティブ実装
function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

function throttle<T extends (...args: any[]) => any>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      fn(...args);
      inThrottle = true;
      setTimeout(() => { inThrottle = false; }, limit);
    }
  };
}

// groupBy のネイティブ実装（Object.groupBy が使える環境）
const grouped = Object.groupBy(users, (user) => user.role);

// uniqBy のネイティブ実装
function uniqBy<T>(arr: T[], key: keyof T): T[] {
  const seen = new Set();
  return arr.filter((item) => {
    const val = item[key];
    if (seen.has(val)) return false;
    seen.add(val);
    return true;
  });
}

// axios → fetch への移行例
// Before
import axios from 'axios';
const { data } = await axios.get('/api/users');
await axios.post('/api/users', { name: 'Taro' });

// After
const data = await fetch('/api/users').then((r) => r.json());
await fetch('/api/users', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ name: 'Taro' }),
});
```

### 1.5 バンドルサイズの目標値

```
推奨バンドルサイズ目標:
  ┌─────────────────────────────────┬─────────────────┬───────────────┐
  │ 指標                             │ 推奨値           │ 警告値         │
  ├─────────────────────────────────┼─────────────────┼───────────────┤
  │ 初期JS（gzip）                   │ < 150KB         │ > 250KB       │
  │ 各ルートのJS（gzip）             │ < 80KB          │ > 150KB       │
  │ First Load JS                   │ < 250KB         │ > 400KB       │
  │ 合計CSS（gzip）                  │ < 50KB          │ > 100KB       │
  │ 最大画像サイズ                    │ < 200KB         │ > 500KB       │
  │ Total Page Weight               │ < 1MB           │ > 2MB         │
  └─────────────────────────────────┴─────────────────┴───────────────┘

ネットワーク別の体感速度:
  ┌──────────────┬──────────┬──────────────────────────────┐
  │ 接続速度      │ 帯域幅    │ 1MBの読み込み時間             │
  ├──────────────┼──────────┼──────────────────────────────┤
  │ 3G           │ 1.5Mbps  │ 約5.3秒                      │
  │ 4G           │ 10Mbps   │ 約0.8秒                      │
  │ 5G           │ 100Mbps  │ 約0.08秒                     │
  │ Wi-Fi        │ 50Mbps   │ 約0.16秒                     │
  └──────────────┴──────────┴──────────────────────────────┘
```

---

## 2. 画像最適化

### 2.1 モダンな画像フォーマット

画像はWebページの総データ量の中で最も大きな割合を占めることが多く、適切なフォーマット選択が重要である。

| フォーマット | 圧縮方式 | 透過 | アニメーション | ブラウザ対応 | ユースケース |
|-----------|---------|-----|-------------|-----------|-----------|
| JPEG | 非可逆 | × | × | 全ブラウザ | 写真、自然画 |
| PNG | 可逆 | ○ | × | 全ブラウザ | ロゴ、スクリーンショット |
| GIF | 可逆 | ○ | ○ | 全ブラウザ | 簡易アニメーション |
| WebP | 可逆/非可逆 | ○ | ○ | 97%+ | 汎用（JPEG/PNGの代替） |
| AVIF | 非可逆 | ○ | ○ | 92%+ | 次世代フォーマット |
| SVG | ベクター | ○ | ○ | 全ブラウザ | アイコン、イラスト |

**フォーマット別の圧縮効率比較（同品質での一般的なファイルサイズ）:**

```
元画像: 1MB JPEG
  → WebP:  約 25-35% 削減 → 650-750KB
  → AVIF:  約 40-60% 削減 → 400-600KB

元画像: 500KB PNG（透過あり）
  → WebP:  約 30-40% 削減 → 300-350KB
  → AVIF:  約 50-70% 削減 → 150-250KB
```

### 2.2 Next.js Image コンポーネント

```typescript
import Image from 'next/image';

// ① 静的画像（ビルド時に最適化）
import heroImage from '@/public/images/hero.jpg';

function HeroSection() {
  return (
    <Image
      src={heroImage}
      alt="メインビジュアル"
      priority               // LCP画像には必ず設定
      placeholder="blur"     // ビルド時にblurDataURLが自動生成
      quality={85}            // 品質（デフォルト: 75）
      sizes="100vw"
    />
  );
}

// ② 動的画像（外部URL）
function UserAvatar({ user }: { user: User }) {
  return (
    <Image
      src={user.avatarUrl}
      alt={`${user.name}のアバター`}
      width={64}
      height={64}
      sizes="64px"
      className="rounded-full"
      // 外部画像のblurプレースホルダー
      placeholder="blur"
      blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    />
  );
}

// ③ レスポンシブ画像（fillモード）
function ProductCard({ product }: { product: Product }) {
  return (
    <div className="relative aspect-[4/3] w-full">
      <Image
        src={product.imageUrl}
        alt={product.name}
        fill
        sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
        className="object-cover rounded-lg"
      />
    </div>
  );
}

// ④ アート・ディレクション（picture要素的な使い方）
function ResponsiveHero() {
  const isMobile = useMediaQuery('(max-width: 768px)');

  return (
    <Image
      src={isMobile ? '/hero-mobile.jpg' : '/hero-desktop.jpg'}
      alt="ヒーローイメージ"
      width={isMobile ? 768 : 1920}
      height={isMobile ? 1024 : 1080}
      priority
      sizes="100vw"
    />
  );
}
```

**next.config.js での画像設定:**

```javascript
// next.config.js
module.exports = {
  images: {
    // 外部画像のドメイン許可
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'images.example.com',
        pathname: '/uploads/**',
      },
      {
        protocol: 'https',
        hostname: '*.cloudinary.com',
      },
    ],
    // 出力フォーマット
    formats: ['image/avif', 'image/webp'],
    // 生成するサイズ一覧
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    // 画像キャッシュのTTL（秒）
    minimumCacheTTL: 60 * 60 * 24 * 30, // 30日
  },
};
```

### 2.3 画像の遅延読み込みとプリロード

```typescript
// ネイティブ lazy loading
<img
  src="/large-image.jpg"
  alt="大きな画像"
  loading="lazy"           // ビューポート外では読み込まない
  decoding="async"         // デコードを非同期で行う
  width="800"
  height="600"
/>

// LCP画像のプリロード（head内に記述）
<link
  rel="preload"
  as="image"
  href="/hero.webp"
  type="image/webp"
  fetchPriority="high"
/>

// fetchpriority による優先度制御
<img
  src="/hero.jpg"
  alt="ヒーロー画像"
  fetchpriority="high"    // LCP画像は高優先度
  width="1920"
  height="1080"
/>

<img
  src="/below-fold.jpg"
  alt="フォールド下の画像"
  fetchpriority="low"     // フォールド下の画像は低優先度
  loading="lazy"
  width="400"
  height="300"
/>
```

### 2.4 画像最適化の自動化パイプライン

```typescript
// sharp を使ったビルド時画像最適化スクリプト
// scripts/optimize-images.ts
import sharp from 'sharp';
import { glob } from 'glob';
import path from 'path';
import fs from 'fs/promises';

interface OptimizeOptions {
  inputDir: string;
  outputDir: string;
  quality: number;
  formats: ('webp' | 'avif')[];
  maxWidth: number;
}

async function optimizeImages(options: OptimizeOptions) {
  const { inputDir, outputDir, quality, formats, maxWidth } = options;

  const images = await glob(`${inputDir}/**/*.{jpg,jpeg,png}`, {});
  console.log(`${images.length} 件の画像を最適化します...`);

  let totalOriginalSize = 0;
  let totalOptimizedSize = 0;

  for (const imagePath of images) {
    const relativePath = path.relative(inputDir, imagePath);
    const baseName = path.basename(relativePath, path.extname(relativePath));
    const dirName = path.dirname(relativePath);
    const outDir = path.join(outputDir, dirName);

    await fs.mkdir(outDir, { recursive: true });

    const originalBuffer = await fs.readFile(imagePath);
    totalOriginalSize += originalBuffer.length;

    const image = sharp(originalBuffer);
    const metadata = await image.metadata();

    // リサイズ（最大幅を超える場合）
    const resizedImage = (metadata.width ?? 0) > maxWidth
      ? image.resize({ width: maxWidth, withoutEnlargement: true })
      : image;

    // 各フォーマットに変換
    for (const format of formats) {
      const outputPath = path.join(outDir, `${baseName}.${format}`);

      if (format === 'webp') {
        const buffer = await resizedImage
          .webp({ quality, effort: 6 })
          .toBuffer();
        await fs.writeFile(outputPath, buffer);
        totalOptimizedSize += buffer.length;
      } else if (format === 'avif') {
        const buffer = await resizedImage
          .avif({ quality, effort: 6 })
          .toBuffer();
        await fs.writeFile(outputPath, buffer);
        totalOptimizedSize += buffer.length;
      }
    }
  }

  const savings = ((1 - totalOptimizedSize / totalOriginalSize) * 100).toFixed(1);
  console.log(`完了: ${savings}% 削減 (${formatBytes(totalOriginalSize)} → ${formatBytes(totalOptimizedSize)})`);
}

function formatBytes(bytes: number): string {
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

// 実行
optimizeImages({
  inputDir: './public/images',
  outputDir: './public/optimized',
  quality: 80,
  formats: ['webp', 'avif'],
  maxWidth: 1920,
});
```

### 2.5 SVG の最適化

```typescript
// SVGO によるSVG最適化
// svgo.config.js
module.exports = {
  plugins: [
    'preset-default',
    'removeDimensions',
    {
      name: 'removeAttrs',
      params: { attrs: '(data-.*)' },
    },
    {
      name: 'addAttributesToSVGElement',
      params: {
        attributes: [{ 'aria-hidden': 'true' }],
      },
    },
  ],
};

// React コンポーネントとしてのSVG（@svgr/webpack）
// webpack / next.config.js
module.exports = {
  webpack(config) {
    config.module.rules.push({
      test: /\.svg$/,
      use: ['@svgr/webpack'],
    });
    return config;
  },
};

// 使い方
import SearchIcon from '@/icons/search.svg';

function SearchButton() {
  return (
    <button aria-label="検索">
      <SearchIcon className="w-5 h-5 text-gray-600" />
    </button>
  );
}
```

---

## 3. フォント最適化

### 3.1 Next.js Font Optimization

Next.js の `next/font` は、フォントファイルをビルド時にダウンロードし、セルフホスティングすることで外部リクエストを排除する。

```typescript
// app/layout.tsx
import { Inter, Noto_Sans_JP } from 'next/font/google';

// 欧文フォント
const inter = Inter({
  subsets: ['latin'],
  display: 'swap',         // FOUT を許容（CLS を防止）
  variable: '--font-inter',
  // 使用するウェイトを限定してサイズ削減
  // weight: ['400', '500', '600', '700'],
  // subsets で必要な文字セットのみ
});

// 日本語フォント
const notoSansJP = Noto_Sans_JP({
  subsets: ['latin'],
  weight: ['400', '500', '700'],  // 必要なウェイトのみ
  display: 'swap',
  variable: '--font-noto',
  preload: false,           // 日本語フォントは大きいのでpreloadしない
  adjustFontFallback: true, // フォールバックフォントのサイズ調整
});

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="ja"
      className={`${inter.variable} ${notoSansJP.variable}`}
    >
      <body className="font-sans">{children}</body>
    </html>
  );
}
```

### 3.2 ローカルフォントの使用

```typescript
// next/font/local の使用
import localFont from 'next/font/local';

const customFont = localFont({
  src: [
    {
      path: '../fonts/CustomFont-Regular.woff2',
      weight: '400',
      style: 'normal',
    },
    {
      path: '../fonts/CustomFont-Bold.woff2',
      weight: '700',
      style: 'normal',
    },
  ],
  display: 'swap',
  variable: '--font-custom',
  // フォントのサブセット化（日本語の場合特に有効）
  // unicode-range で必要な文字のみ
});
```

### 3.3 font-display 戦略の比較

```
font-display の各値と挙動:

  ┌──────────┬──────────────────────────┬──────────────┬──────────────┐
  │ 値        │ 動作                      │ CLS影響      │ ユースケース  │
  ├──────────┼──────────────────────────┼──────────────┼──────────────┤
  │ swap     │ フォールバックを即表示、    │ 中（FOUT）   │ 本文テキスト  │
  │          │ ロード後に切り替え          │              │              │
  ├──────────┼──────────────────────────┼──────────────┼──────────────┤
  │ block    │ 3秒間非表示、              │ 高（FOIT）   │ アイコン     │
  │          │ その後フォールバック        │              │ フォント     │
  ├──────────┼──────────────────────────┼──────────────┼──────────────┤
  │ fallback │ 100msの非表示期間後に      │ 低           │ バランス型   │
  │          │ フォールバック、3秒以内     │              │              │
  │          │ にロードできれば切り替え    │              │              │
  ├──────────┼──────────────────────────┼──────────────┼──────────────┤
  │ optional │ 100msの非表示期間後に      │ なし         │ 重要でない   │
  │          │ フォールバック、次回訪問    │              │ テキスト     │
  │          │ から使用                   │              │              │
  └──────────┴──────────────────────────┴──────────────┴──────────────┘

推奨: Core Web Vitals を重視する場合は swap または optional
```

### 3.4 フォントサブセット化

```bash
# pyftsubset を使った日本語フォントのサブセット化
pip install fonttools brotli

# 第一水準漢字 + ひらがな + カタカナ + 記号のみに削減
pyftsubset NotoSansJP-Regular.ttf \
  --text-file=characters.txt \
  --output-file=NotoSansJP-Regular-subset.woff2 \
  --flavor=woff2 \
  --layout-features='*'

# unicode-range による段階的読み込み
# @font-face で文字範囲ごとに分割
```

```css
/* CSS での unicode-range によるフォント分割 */
/* ラテン文字 */
@font-face {
  font-family: 'NotoSansJP';
  font-weight: 400;
  font-display: swap;
  src: url('/fonts/NotoSansJP-Regular-latin.woff2') format('woff2');
  unicode-range: U+0000-007F, U+2000-206F;
}

/* ひらがな・カタカナ */
@font-face {
  font-family: 'NotoSansJP';
  font-weight: 400;
  font-display: swap;
  src: url('/fonts/NotoSansJP-Regular-kana.woff2') format('woff2');
  unicode-range: U+3000-30FF, U+FF00-FFEF;
}

/* 漢字（第一水準） */
@font-face {
  font-family: 'NotoSansJP';
  font-weight: 400;
  font-display: swap;
  src: url('/fonts/NotoSansJP-Regular-kanji.woff2') format('woff2');
  unicode-range: U+4E00-9FFF;
}
```

### 3.5 Tailwind CSS でのフォント設定

```typescript
// tailwind.config.ts
import type { Config } from 'tailwindcss';

const config: Config = {
  theme: {
    extend: {
      fontFamily: {
        sans: [
          'var(--font-inter)',
          'var(--font-noto)',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'sans-serif',
        ],
        mono: [
          'var(--font-jetbrains-mono)',
          'ui-monospace',
          'SFMono-Regular',
          'monospace',
        ],
      },
    },
  },
};

export default config;
```

---

## 4. CSS 最適化

### 4.1 未使用CSSの除去

```javascript
// Tailwind CSS の purge（content 設定）
// tailwind.config.ts
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './lib/**/*.{js,ts,jsx,tsx}',
    // サードパーティコンポーネントも含める
    './node_modules/@company/ui/**/*.{js,ts,jsx,tsx}',
  ],
  // safelist: 動的に生成されるクラスを保護
  safelist: [
    'bg-red-500',
    'bg-green-500',
    'bg-blue-500',
    { pattern: /^bg-(red|green|blue)-(100|500|900)$/ },
  ],
};

// PurgeCSS の単体使用（Tailwind以外のプロジェクト）
// postcss.config.js
const purgecss = require('@fullhuman/postcss-purgecss');

module.exports = {
  plugins: [
    require('tailwindcss'),
    require('autoprefixer'),
    ...(process.env.NODE_ENV === 'production'
      ? [purgecss({
          content: ['./src/**/*.{js,jsx,ts,tsx,html}'],
          defaultExtractor: (content) =>
            content.match(/[\w-/:]+(?<!:)/g) || [],
          safelist: {
            standard: [/^data-/, /^aria-/],
            deep: [/modal/, /tooltip/],
          },
        })]
      : []),
  ],
};
```

### 4.2 Critical CSS のインライン化

```typescript
// critters による Critical CSS の自動抽出（Next.jsは内蔵）
// next.config.js
module.exports = {
  experimental: {
    optimizeCss: true, // Critical CSSの自動インライン化
  },
};

// 手動での Critical CSS 戦略
// <head>内にAbove-the-fold CSSをインライン化
// 残りのCSSは非同期読み込み
```

```html
<!-- Critical CSS のインライン化パターン -->
<head>
  <!-- クリティカルCSSをインライン -->
  <style>
    /* ファーストビューに必要な最小限のCSS */
    :root { --primary: #3b82f6; }
    body { margin: 0; font-family: system-ui, sans-serif; }
    .hero { min-height: 100vh; display: flex; align-items: center; }
    .nav { position: fixed; top: 0; width: 100%; z-index: 50; }
  </style>

  <!-- 残りのCSSは非同期で読み込み -->
  <link
    rel="preload"
    href="/styles/main.css"
    as="style"
    onload="this.onload=null;this.rel='stylesheet'"
  />
  <noscript>
    <link rel="stylesheet" href="/styles/main.css" />
  </noscript>
</head>
```

### 4.3 CSS-in-JS のパフォーマンス考慮

```typescript
// ランタイムCSS-in-JS のパフォーマンス問題
// styled-components, emotion はランタイムにCSSを生成する

// 推奨: ゼロランタイムCSS-in-JS への移行
// - Tailwind CSS（ユーティリティファースト）
// - CSS Modules（ビルド時にスコープ化）
// - vanilla-extract（ゼロランタイム、TypeScript対応）
// - Panda CSS（ゼロランタイム、Tailwind風API）

// vanilla-extract の例
// styles.css.ts
import { style, globalStyle } from '@vanilla-extract/css';

export const container = style({
  maxWidth: '1200px',
  margin: '0 auto',
  padding: '0 1rem',
  '@media': {
    '(min-width: 768px)': {
      padding: '0 2rem',
    },
  },
});

export const heading = style({
  fontSize: '2rem',
  fontWeight: 700,
  color: '#1a1a1a',
});
```

---

## 5. キャッシュ戦略

### 5.1 Next.js のキャッシュ階層

Next.js App Router には4層のキャッシュ機構が存在し、それぞれの特性を理解することが重要である。

```
Next.js キャッシュ階層の全体像:

  リクエスト
    │
    ▼
  ① Router Cache（クライアント側）
    │  ・ブラウザのメモリに保持
    │  ・prefetch されたルートをキャッシュ
    │  ・セッション間では持続しない
    │
    ▼
  ② Full Route Cache（サーバー側）
    │  ・静的にレンダリングされたルートのHTML + RSC Payload
    │  ・ビルド時または revalidate 後に生成
    │  ・CDN でキャッシュ可能
    │
    ▼
  ③ Data Cache（サーバー側）
    │  ・fetch() の結果をキャッシュ
    │  ・デプロイ間で持続する
    │  ・revalidate で有効期限を設定
    │
    ▼
  ④ Request Memoization（サーバー側）
     ・同一レンダリングパス内の重複 fetch を自動排除
     ・React の機能（fetch の自動メモ化）
     ・リクエスト完了後に破棄
```

### 5.2 Data Cache の制御

```typescript
// ① キャッシュなし（毎回フェッチ）
async function getLatestData() {
  const res = await fetch('https://api.example.com/data', {
    cache: 'no-store',
  });
  return res.json();
}

// ② 時間ベースの再検証（ISR的）
async function getProducts() {
  const res = await fetch('https://api.example.com/products', {
    next: { revalidate: 60 }, // 60秒間キャッシュ
  });
  return res.json();
}

// ③ タグベースの再検証
async function getBlogPosts() {
  const res = await fetch('https://api.example.com/posts', {
    next: { tags: ['posts'] }, // 'posts'タグでグループ化
  });
  return res.json();
}

// ④ ページレベルのキャッシュ制御
// app/products/page.tsx
export const revalidate = 60;           // 60秒間キャッシュ
export const dynamic = 'force-dynamic'; // SSR強制（キャッシュなし）
export const dynamic = 'force-static';  // 静的生成を強制
export const fetchCache = 'default-no-store'; // デフォルトキャッシュなし

// ⑤ オンデマンドでのキャッシュ無効化
import { revalidatePath, revalidateTag } from 'next/cache';

// Server Action / Route Handler 内で使用
export async function createPost(formData: FormData) {
  'use server';

  await db.post.create({ ... });

  // パスベースの無効化
  revalidatePath('/blog');          // /blog ページを再生成
  revalidatePath('/blog', 'layout'); // /blog 配下すべて

  // タグベースの無効化
  revalidateTag('posts');           // 'posts'タグのキャッシュを無効化
}
```

### 5.3 HTTPキャッシュヘッダー

```typescript
// Next.js Route Handler でのキャッシュヘッダー設定
// app/api/data/route.ts
import { NextResponse } from 'next/server';

export async function GET() {
  const data = await fetchData();

  return NextResponse.json(data, {
    headers: {
      // 公開キャッシュ: CDN + ブラウザで60秒キャッシュ
      // stale-while-revalidate: 期限切れ後も古いデータを返しつつ裏で更新
      'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=300',
    },
  });
}

// 静的アセットのキャッシュ戦略
// next.config.js
module.exports = {
  async headers() {
    return [
      {
        // 静的アセット（ハッシュ付きファイル名）
        source: '/_next/static/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      {
        // 画像
        source: '/images/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=86400, stale-while-revalidate=604800',
          },
        ],
      },
      {
        // APIレスポンス
        source: '/api/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, s-maxage=60, stale-while-revalidate=300',
          },
        ],
      },
    ];
  },
};
```

### 5.4 Cache-Control ディレクティブ一覧

```
Cache-Control ディレクティブの詳細:

  ┌──────────────────────────┬────────────────────────────────────────┐
  │ ディレクティブ            │ 説明                                    │
  ├──────────────────────────┼────────────────────────────────────────┤
  │ public                   │ CDN・共有キャッシュに保存可能             │
  │ private                  │ ブラウザのみ（CDNに保存しない）           │
  │ no-cache                 │ 毎回サーバーに検証（304応答可）           │
  │ no-store                 │ 一切キャッシュしない                      │
  │ max-age=N                │ N秒間キャッシュ有効                      │
  │ s-maxage=N               │ 共有キャッシュ（CDN）でのmax-age         │
  │ stale-while-revalidate=N │ 期限切れ後もN秒間は古いデータを返す       │
  │ stale-if-error=N         │ エラー時にN秒間は古いデータを返す         │
  │ immutable                │ コンテンツが変更されないことを宣言         │
  │ must-revalidate          │ 期限切れ後は必ずサーバーに問い合わせ       │
  └──────────────────────────┴────────────────────────────────────────┘

よくある設定パターン:
  静的アセット（JS/CSS/画像、ハッシュ付き）:
    Cache-Control: public, max-age=31536000, immutable

  HTML（頻繁に更新）:
    Cache-Control: no-cache

  API（短期キャッシュ）:
    Cache-Control: public, s-maxage=60, stale-while-revalidate=300

  個人データ:
    Cache-Control: private, no-cache

  認証済みページ:
    Cache-Control: private, no-store
```

### 5.5 Service Worker によるキャッシュ

```typescript
// next-pwa を使った Service Worker キャッシュ
// next.config.js
const withPWA = require('next-pwa')({
  dest: 'public',
  disable: process.env.NODE_ENV === 'development',
  runtimeCaching: [
    {
      // 画像のキャッシュ
      urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp|avif)$/i,
      handler: 'CacheFirst',
      options: {
        cacheName: 'images',
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 60 * 60 * 24 * 30, // 30日
        },
      },
    },
    {
      // API レスポンスのキャッシュ
      urlPattern: /\/api\/.*/i,
      handler: 'NetworkFirst',
      options: {
        cacheName: 'api-cache',
        expiration: {
          maxEntries: 50,
          maxAgeSeconds: 60 * 5, // 5分
        },
        networkTimeoutSeconds: 10,
      },
    },
    {
      // フォントのキャッシュ
      urlPattern: /\.(?:woff|woff2|ttf|otf)$/i,
      handler: 'CacheFirst',
      options: {
        cacheName: 'fonts',
        expiration: {
          maxEntries: 10,
          maxAgeSeconds: 60 * 60 * 24 * 365, // 1年
        },
      },
    },
  ],
});

module.exports = withPWA({
  // Next.js config
});
```

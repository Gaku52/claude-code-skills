# プロジェクト構成

> プロジェクト構成は開発チームの生産性を決定づける。Feature-based構成、レイヤードアーキテクチャ、モノレポ設計まで、スケーラブルで保守しやすいディレクトリ設計の原則とパターンを習得する。

## 前提知識

この章を学ぶ前に、以下の知識を習得しておくことを推奨する。

- SPA/MPA/SSRの概念とレンダリング方式の違い
  - 参照: `./00-spa-mpa-ssr.md`
- モジュールシステムの理解（ESM vs CommonJS）
  - import/export の仕組み
  - Tree-shaking の原理
  - Dynamic import による遅延読み込み

## この章で学ぶこと

- [ ] Feature-basedなディレクトリ設計を理解する
- [ ] レイヤードアーキテクチャの適用方法を把握する
- [ ] モノレポとパッケージ分割の設計を学ぶ
- [ ] ESLint による依存ルールの自動化を習得する
- [ ] テスト配置戦略とCI/CDへの統合を理解する
- [ ] プロジェクト規模に応じた構成の選択基準を把握する

## 前提知識

この章を学習する前に、以下の知識を習得しておくことを推奨します。

- **SPA/MPA/SSRの概念**: 各レンダリング方式の特徴と選定基準の理解
  → 参照: `./00-spa-mpa-ssr.md`
- **モジュールシステム**: ESM（ES Modules）とCommonJSの違い、import/export の仕組み
- **パス解決**: 相対パス vs 絶対パス、TypeScriptのpaths設定の理解

---

## 1. ディレクトリ設計の原則

### 1.1 なぜディレクトリ設計が重要なのか

ディレクトリ設計はソフトウェアアーキテクチャの「第一印象」であり、プロジェクト全体の品質に直結する。新しいメンバーがリポジトリを開いた瞬間、コードの場所が予測でき、修正すべきファイルに迷わず辿り着ける構成が理想的である。

**良いディレクトリ設計がもたらす効果:**

| 効果 | 説明 |
|------|------|
| 発見性（Discoverability） | 「このコードはどこにあるか？」が直感的にわかる |
| 変更容易性（Changeability） | 機能追加・修正の影響範囲が明確で局所化される |
| 削除容易性（Deletability） | 不要な機能をディレクトリごと安全に削除できる |
| チームスケール | 複数チームが独立して作業でき、コンフリクトが減少する |
| オンボーディング速度 | 新メンバーが短時間でプロジェクト全体を把握できる |
| テスト効率 | 変更された機能に対するテストのみを実行できる |

### 1.2 アンチパターン: 型ベース（Technical-based）構成

多くのプロジェクトが最初に採用しがちな構成が「型ベース」のディレクトリ設計である。これは技術的な役割でファイルをグループ化する。

```
型ベースの構成（アンチパターン）:

  src/
  ├── components/         ← 全コンポーネントが混在
  │   ├── Button.tsx
  │   ├── UserList.tsx
  │   ├── UserCard.tsx
  │   ├── OrderTable.tsx
  │   ├── OrderStatusBadge.tsx
  │   ├── AuthForm.tsx
  │   ├── SettingsPanel.tsx
  │   ├── DashboardChart.tsx
  │   └── ... (100+ ファイル)
  ├── hooks/              ← 全hooksが混在
  │   ├── useUsers.ts
  │   ├── useOrders.ts
  │   ├── useAuth.ts
  │   ├── useLocalStorage.ts
  │   └── ... (50+ ファイル)
  ├── utils/              ← 全ユーティリティが混在
  │   ├── formatDate.ts
  │   ├── validateEmail.ts
  │   ├── calculatePrice.ts
  │   └── ... (30+ ファイル)
  ├── types/              ← 全型が混在
  │   ├── user.ts
  │   ├── order.ts
  │   ├── auth.ts
  │   └── ...
  ├── api/                ← 全API呼び出しが混在
  │   ├── users.ts
  │   ├── orders.ts
  │   └── auth.ts
  └── constants/
      ├── userRoles.ts
      └── orderStatus.ts
```

**型ベース構成の問題点:**

```
問題1: スケーラビリティの欠如
  - components/ に100個以上のファイルが並び、目的のファイルを見つけられない
  - ファイル名でアルファベット順にソートしても関連性は見えない
  - IDE のファイルツリーをスクロールし続ける必要がある

問題2: 関連コードの分散
  - User機能に関連するコードが5箇所以上に分散
    components/UserList.tsx, hooks/useUsers.ts, api/users.ts,
    types/user.ts, utils/formatUser.ts
  - 1つの機能を変更するために複数ディレクトリを行き来する

問題3: 依存関係の不透明性
  - どのコンポーネントがどのhookを使っているか把握困難
  - 循環参照が発生しやすい
  - 影響範囲の分析ができない

問題4: 削除・リファクタリングの困難
  - 機能を削除するとき、関連ファイルが散在しているため漏れが生じる
  - 「このユーティリティは他で使われているか？」の判断が難しい
  - Dead code が蓄積しやすい
```

### 1.3 推奨パターン: Feature-based 構成

Feature-based構成は、ビジネスドメインの機能単位でコードをグループ化する。

```
Feature-based 構成（推奨）:

  src/
  ├── features/           ← 機能ごとにまとめる
  │   ├── users/
  │   │   ├── components/
  │   │   │   ├── UserList.tsx
  │   │   │   ├── UserCard.tsx
  │   │   │   └── UserSearchInput.tsx
  │   │   ├── hooks/
  │   │   │   ├── useUsers.ts
  │   │   │   └── useUser.ts
  │   │   ├── api/
  │   │   │   ├── queries.ts
  │   │   │   └── actions.ts
  │   │   ├── types/
  │   │   │   └── user.ts
  │   │   ├── utils/
  │   │   │   └── format.ts
  │   │   ├── __tests__/
  │   │   │   ├── UserList.test.tsx
  │   │   │   └── useUsers.test.ts
  │   │   └── index.ts    ← 公開APIの定義
  │   ├── orders/
  │   │   ├── components/
  │   │   ├── hooks/
  │   │   ├── api/
  │   │   ├── types/
  │   │   ├── __tests__/
  │   │   └── index.ts
  │   ├── auth/
  │   │   ├── components/
  │   │   ├── hooks/
  │   │   ├── providers/
  │   │   └── index.ts
  │   └── notifications/
  │       ├── components/
  │       ├── hooks/
  │       └── index.ts
  ├── shared/             ← 共有コンポーネント・ユーティリティ
  │   ├── components/
  │   │   ├── ui/
  │   │   │   ├── Button.tsx
  │   │   │   ├── Modal.tsx
  │   │   │   ├── Table.tsx
  │   │   │   └── Input.tsx
  │   │   └── layout/
  │   │       ├── Header.tsx
  │   │       ├── Sidebar.tsx
  │   │       └── Footer.tsx
  │   ├── hooks/
  │   │   ├── useLocalStorage.ts
  │   │   ├── useDebounce.ts
  │   │   └── useMediaQuery.ts
  │   ├── lib/
  │   │   ├── api-client.ts
  │   │   ├── auth.ts
  │   │   └── utils.ts
  │   ├── types/
  │   │   ├── api.ts
  │   │   └── common.ts
  │   └── constants/
  │       └── routes.ts
  └── app/                ← ルーティング・レイアウト
      ├── layout.tsx
      ├── page.tsx
      └── (routes)/
```

**Feature-based 構成の利点:**

| 利点 | 詳細 |
|------|------|
| 高凝集 | 関連するコードが1ディレクトリに集約される |
| 低結合 | Feature間の依存を明示的に制御できる |
| 独立デプロイ | 将来的なマイクロフロントエンド化が容易 |
| チーム分割 | Feature単位でオーナーシップを設定できる |
| テスト容易性 | Feature単位でテストを実行・管理できる |

### 1.4 Feature-based 構成への段階的移行

既存の型ベースプロジェクトからFeature-basedへ移行する手順は以下のとおりである。

```typescript
// Step 1: shared/ ディレクトリを作成し、汎用コードを移動
// 移行前
// src/components/Button.tsx → src/shared/components/ui/Button.tsx
// src/hooks/useLocalStorage.ts → src/shared/hooks/useLocalStorage.ts

// Step 2: 最もまとまりのある機能から features/ に切り出す
// src/components/UserList.tsx → src/features/users/components/UserList.tsx
// src/hooks/useUsers.ts → src/features/users/hooks/useUsers.ts
// src/api/users.ts → src/features/users/api/queries.ts
// src/types/user.ts → src/features/users/types/user.ts

// Step 3: index.ts を作成して公開APIを定義
// src/features/users/index.ts
export { UserList } from './components/UserList';
export { useUsers } from './hooks/useUsers';
export type { User } from './types/user';

// Step 4: 既存のimportを更新
// Before
import { UserList } from '@/components/UserList';
import { useUsers } from '@/hooks/useUsers';

// After
import { UserList, useUsers } from '@/features/users';

// Step 5: 残りの機能を順次移行
// orders, auth, notifications, ...
```

**移行時の注意点:**

```
1. 一度に全部移行しない
   - 1つの feature ずつ移行し、各段階でテストを実行
   - PR は feature 単位で作成

2. import の自動更新ツールを活用
   - VS Code の "Move to a new file" 機能
   - TypeScript の Language Service が参照を更新
   - jscodeshift を使ったバッチ処理

3. CI でのインポートルールチェック
   - ESLint の import/no-restricted-paths を設定
   - 移行完了した feature への旧パスからのアクセスを禁止

4. ドキュメントの整備
   - ARCHITECTURE.md に新しい構成ルールを記載
   - ADR（Architecture Decision Records）に移行の判断理由を記録
```

### 1.5 構成パターンの比較

プロジェクトの規模やチーム構成に応じて適切なパターンは異なる。

```
パターン比較表:

  パターン          | 適用規模    | メリット              | デメリット
  ─────────────────|───────────|─────────────────────|───────────────────
  型ベース          | 小規模     | シンプル、学習コスト低  | スケールしない
  Feature-based    | 中〜大規模  | 高凝集、低結合         | 初期設計コスト
  レイヤード        | 中規模     | 責務が明確            | 横断的変更が多い
  モジュラーモノリス | 大規模     | マイクロサービス準備   | 設計スキル必要
  マイクロフロントエンド | 超大規模 | チーム完全独立        | 運用コスト高い

  選択基準:
  - ファイル数 < 50    → 型ベースでも可
  - ファイル数 50-200  → Feature-based 推奨
  - ファイル数 200+    → Feature-based + モノレポ
  - チーム 1-3人       → Feature-based
  - チーム 4-10人      → Feature-based + strict import rules
  - チーム 10+人       → モノレポ + Feature-based
```

---

## 2. Next.js App Router のプロジェクト構成

### 2.1 基本構成

Next.js 14+ の App Router を使用する場合の推奨構成を示す。App Router はファイルシステムベースのルーティングを採用しており、ディレクトリ構成がそのままURLに対応する。

```
推奨構成（Next.js 14+）:

  project-root/
  ├── src/
  │   ├── app/                    ← ルーティング（App Router）
  │   │   ├── layout.tsx          ← ルートレイアウト
  │   │   ├── page.tsx            ← / ページ
  │   │   ├── error.tsx           ← グローバルエラーUI
  │   │   ├── loading.tsx         ← グローバルローディングUI
  │   │   ├── not-found.tsx       ← 404ページ
  │   │   ├── global-error.tsx    ← ルートエラーバウンダリ
  │   │   ├── (marketing)/        ← ルートグループ（URLに影響なし）
  │   │   │   ├── layout.tsx      ← マーケティング用レイアウト
  │   │   │   ├── page.tsx        ← /
  │   │   │   ├── about/
  │   │   │   │   └── page.tsx    ← /about
  │   │   │   ├── pricing/
  │   │   │   │   └── page.tsx    ← /pricing
  │   │   │   └── blog/
  │   │   │       ├── page.tsx    ← /blog
  │   │   │       └── [slug]/
  │   │   │           └── page.tsx ← /blog/:slug
  │   │   ├── (app)/              ← 認証必要エリア
  │   │   │   ├── layout.tsx      ← 認証チェック付きレイアウト
  │   │   │   ├── dashboard/
  │   │   │   │   ├── page.tsx    ← /dashboard
  │   │   │   │   └── loading.tsx ← ダッシュボード用ローディング
  │   │   │   ├── settings/
  │   │   │   │   ├── page.tsx    ← /settings
  │   │   │   │   ├── profile/
  │   │   │   │   │   └── page.tsx
  │   │   │   │   └── billing/
  │   │   │   │       └── page.tsx
  │   │   │   └── projects/
  │   │   │       ├── page.tsx    ← /projects
  │   │   │       ├── new/
  │   │   │       │   └── page.tsx ← /projects/new
  │   │   │       └── [id]/
  │   │   │           ├── page.tsx ← /projects/:id
  │   │   │           └── edit/
  │   │   │               └── page.tsx
  │   │   └── api/                ← API Routes (Route Handlers)
  │   │       ├── auth/
  │   │       │   └── [...nextauth]/
  │   │       │       └── route.ts
  │   │       └── webhooks/
  │   │           └── stripe/
  │   │               └── route.ts
  │   ├── features/               ← 機能モジュール
  │   │   ├── users/
  │   │   ├── projects/
  │   │   ├── billing/
  │   │   ├── auth/
  │   │   └── notifications/
  │   ├── shared/                 ← 共有リソース
  │   │   ├── components/
  │   │   │   ├── ui/             ← shadcn/ui等の基本UI
  │   │   │   └── layout/         ← レイアウトコンポーネント
  │   │   ├── hooks/
  │   │   ├── lib/                ← ユーティリティ
  │   │   │   ├── api-client.ts
  │   │   │   ├── auth.ts
  │   │   │   ├── db.ts
  │   │   │   └── utils.ts
  │   │   ├── types/
  │   │   └── constants/
  │   ├── middleware.ts            ← Next.js Middleware
  │   └── styles/
  │       └── globals.css
  ├── public/
  │   ├── images/
  │   ├── fonts/
  │   └── favicon.ico
  ├── prisma/                     ← Prisma スキーマ
  │   ├── schema.prisma
  │   ├── seed.ts
  │   └── migrations/
  ├── tests/
  │   ├── e2e/                    ← E2Eテスト（Playwright）
  │   └── integration/            ← 統合テスト
  ├── .github/
  │   └── workflows/
  ├── next.config.js
  ├── tailwind.config.ts
  ├── tsconfig.json
  ├── .env.local
  ├── .env.example
  └── package.json
```

### 2.2 App Router の特殊ファイル

Next.js App Router では、特定のファイル名が特別な意味を持つ。

```typescript
// --- layout.tsx ---
// ページ間で共有されるレイアウト。子ルートがnavigation しても再レンダリングされない
// src/app/(app)/layout.tsx
import { redirect } from 'next/navigation';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/shared/lib/auth';
import { Sidebar } from '@/shared/components/layout/Sidebar';
import { Header } from '@/shared/components/layout/Header';

export default async function AppLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getServerSession(authOptions);

  if (!session) {
    redirect('/login');
  }

  return (
    <div className="flex h-screen">
      <Sidebar user={session.user} />
      <div className="flex-1 flex flex-col">
        <Header user={session.user} />
        <main className="flex-1 overflow-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
}

// --- page.tsx ---
// ルートのUIを定義。このファイルがあるディレクトリがURLに対応
// src/app/(app)/dashboard/page.tsx
import { Suspense } from 'react';
import { DashboardStats } from '@/features/dashboard/components/DashboardStats';
import { RecentActivity } from '@/features/dashboard/components/RecentActivity';
import { DashboardSkeleton } from '@/features/dashboard/components/DashboardSkeleton';

export const metadata = {
  title: 'Dashboard | MyApp',
  description: 'Your dashboard overview',
};

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Dashboard</h1>
      <Suspense fallback={<DashboardSkeleton />}>
        <DashboardStats />
      </Suspense>
      <Suspense fallback={<div>Loading activity...</div>}>
        <RecentActivity />
      </Suspense>
    </div>
  );
}

// --- error.tsx ---
// エラーバウンダリ。実行時エラーをキャッチして fallback UI を表示
// src/app/(app)/dashboard/error.tsx
'use client'; // Error components must be Client Components

import { useEffect } from 'react';
import { Button } from '@/shared/components/ui/Button';

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // エラー報告サービスにログ送信
    console.error('Dashboard error:', error);
  }, [error]);

  return (
    <div className="flex flex-col items-center justify-center gap-4 py-12">
      <h2 className="text-xl font-semibold">Something went wrong!</h2>
      <p className="text-muted-foreground">{error.message}</p>
      <Button onClick={() => reset()}>Try again</Button>
    </div>
  );
}

// --- loading.tsx ---
// Suspense 境界のフォールバックUI。ストリーミングSSRで活用
// src/app/(app)/dashboard/loading.tsx
import { Skeleton } from '@/shared/components/ui/Skeleton';

export default function DashboardLoading() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-10 w-48" />
      <div className="grid grid-cols-3 gap-4">
        <Skeleton className="h-32" />
        <Skeleton className="h-32" />
        <Skeleton className="h-32" />
      </div>
      <Skeleton className="h-64" />
    </div>
  );
}

// --- not-found.tsx ---
// カスタム404ページ
// src/app/not-found.tsx
import Link from 'next/link';
import { Button } from '@/shared/components/ui/Button';

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-4">
      <h1 className="text-6xl font-bold">404</h1>
      <p className="text-xl text-muted-foreground">Page not found</p>
      <Button asChild>
        <Link href="/">Go Home</Link>
      </Button>
    </div>
  );
}

// --- route.ts ---
// API Route Handler。RESTful なAPIエンドポイントを定義
// src/app/api/webhooks/stripe/route.ts
import { headers } from 'next/headers';
import { NextResponse } from 'next/server';
import Stripe from 'stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!);

export async function POST(request: Request) {
  const body = await request.text();
  const signature = headers().get('stripe-signature')!;

  try {
    const event = stripe.webhooks.constructEvent(
      body,
      signature,
      process.env.STRIPE_WEBHOOK_SECRET!
    );

    switch (event.type) {
      case 'checkout.session.completed':
        // 支払い完了処理
        break;
      case 'customer.subscription.updated':
        // サブスクリプション更新処理
        break;
    }

    return NextResponse.json({ received: true });
  } catch (err) {
    return NextResponse.json(
      { error: 'Webhook signature verification failed' },
      { status: 400 }
    );
  }
}
```

### 2.3 ルートグループの活用

ルートグループ `()` はURLに影響を与えずにルートを論理的にグループ化する。

```
ルートグループの活用パターン:

  src/app/
  ├── (marketing)/          ← 公開ページ（ヘッダー・フッター付き）
  │   ├── layout.tsx        ← マーケティング用レイアウト
  │   ├── page.tsx
  │   ├── about/
  │   └── pricing/
  ├── (app)/                ← 認証が必要なページ（サイドバー付き）
  │   ├── layout.tsx        ← アプリ用レイアウト（認証チェック）
  │   ├── dashboard/
  │   └── settings/
  ├── (auth)/               ← 認証関連ページ（ミニマルレイアウト）
  │   ├── layout.tsx        ← 認証用レイアウト
  │   ├── login/
  │   ├── register/
  │   └── forgot-password/
  └── layout.tsx            ← ルートレイアウト（Providers等）

  利点:
  ✓ レイアウトをグループごとに分離できる
  ✓ URLパスに影響を与えない
  ✓ 認証・非認証エリアを明確に分離
  ✓ CSS やロジックのスコープを限定
```

```typescript
// src/app/(auth)/layout.tsx — 認証ページ用のミニマルレイアウト
export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full">
        {children}
      </div>
    </div>
  );
}

// src/app/(marketing)/layout.tsx — マーケティング用レイアウト
import { MarketingHeader } from '@/shared/components/layout/MarketingHeader';
import { Footer } from '@/shared/components/layout/Footer';

export default function MarketingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <MarketingHeader />
      <main>{children}</main>
      <Footer />
    </>
  );
}
```

### 2.4 app/ ディレクトリの設計ルール

```
  ルール1: app/ にはルーティングとページコンポーネントのみ
    app/ 内のファイルは「どのURLでどのコンポーネントを表示するか」のみを担当
    ビジネスロジック、データ取得、状態管理は features/ に委譲

  ルール2: ビジネスロジックは features/ に配置
    ページコンポーネントは features/ のコンポーネントを組み合わせるだけ
    直接 DB クエリや API 呼び出しをページファイル内に書かない

  ルール3: 2つ以上の feature で使うものは shared/ に
    Button, Modal, Table などの汎用UIは shared/components/
    useLocalStorage, useDebounce などの汎用hooksは shared/hooks/
    日付フォーマット、バリデーションなどは shared/lib/

  ルール4: features間の直接import は禁止（shared 経由）
    features/users/ → features/orders/ は NG
    代わりに shared/ にインターフェースを定義して経由する
```

```typescript
// ルール違反の例（BAD）
// src/app/(app)/dashboard/page.tsx
import { prisma } from '@/shared/lib/db';  // ← ページ内でDB直接アクセス

export default async function DashboardPage() {
  const users = await prisma.user.findMany();  // ← ビジネスロジックがページに混入
  const orders = await prisma.order.findMany({
    where: { status: 'pending' },
    include: { user: true },
  });

  return (
    <div>
      <h1>Dashboard</h1>
      {/* 直接データを表示 */}
      <ul>{users.map(u => <li key={u.id}>{u.name}</li>)}</ul>
    </div>
  );
}

// ルール準拠の例（GOOD）
// src/app/(app)/dashboard/page.tsx
import { DashboardStats } from '@/features/dashboard/components/DashboardStats';
import { RecentOrders } from '@/features/orders/components/RecentOrders';
import { ActiveUsers } from '@/features/users/components/ActiveUsers';

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Dashboard</h1>
      <DashboardStats />
      <div className="grid grid-cols-2 gap-6">
        <RecentOrders />
        <ActiveUsers />
      </div>
    </div>
  );
}

// データ取得は features/ 内で行う
// src/features/dashboard/components/DashboardStats.tsx
import { getDashboardStats } from '../api/queries';

export async function DashboardStats() {
  const stats = await getDashboardStats();

  return (
    <div className="grid grid-cols-4 gap-4">
      <StatCard title="Total Users" value={stats.totalUsers} />
      <StatCard title="Active Orders" value={stats.activeOrders} />
      <StatCard title="Revenue" value={`$${stats.revenue}`} />
      <StatCard title="Conversion" value={`${stats.conversionRate}%`} />
    </div>
  );
}
```

---

## 3. Feature モジュールの設計

### 3.1 Feature モジュールの構造

各 feature モジュールは自己完結した単位であり、明確な内部構造を持つ。

```typescript
// features/users/ の内部構造
// features/users/
// ├── components/           ← UIコンポーネント
// │   ├── UserList.tsx      ← Server Component（データフェッチ含む）
// │   ├── UserCard.tsx      ← Server Component
// │   ├── UserAvatar.tsx    ← Server Component
// │   ├── UserSearchInput.tsx ← Client Component（インタラクティブ）
// │   └── UserForm.tsx      ← Client Component（フォーム）
// ├── hooks/                ← カスタムhooks
// │   ├── useUsers.ts       ← ユーザー一覧取得
// │   ├── useUser.ts        ← 個別ユーザー取得
// │   └── useUserForm.ts    ← フォームロジック
// ├── api/                  ← データ取得・更新
// │   ├── queries.ts        ← サーバーサイドクエリ / TanStack Query
// │   └── actions.ts        ← Server Actions
// ├── types/                ← 型定義
// │   └── user.ts
// ├── utils/                ← feature固有のユーティリティ
// │   ├── format.ts         ← ユーザー名フォーマット等
// │   └── validation.ts     ← バリデーションスキーマ
// ├── constants/            ← feature固有の定数
// │   └── roles.ts
// ├── __tests__/            ← テスト
// │   ├── UserList.test.tsx
// │   ├── useUsers.test.ts
// │   └── actions.test.ts
// └── index.ts              ← 公開API

// features/users/index.ts — 公開API定義
// このファイルからのみ外部アクセスが可能

// Components（外部から使用を許可するもののみ）
export { UserList } from './components/UserList';
export { UserCard } from './components/UserCard';
export { UserAvatar } from './components/UserAvatar';

// Hooks
export { useUsers } from './hooks/useUsers';
export { useUser } from './hooks/useUser';

// Types
export type { User, CreateUserInput, UpdateUserInput } from './types/user';

// Actions
export { createUser, updateUser, deleteUser } from './api/actions';

// 注意: 内部実装は export しない
// UserSearchInput, UserForm は内部コンポーネントとして扱う
// format.ts, validation.ts も外部には公開しない
```

### 3.2 API レイヤーの設計

```typescript
// features/users/api/queries.ts
// サーバーサイドでのデータ取得（Server Components から呼び出す）
import { prisma } from '@/shared/lib/db';
import { cache } from 'react';

// React の cache() で同一レンダリング内の重複リクエストを排除
export const getUsers = cache(async (params?: {
  page?: number;
  limit?: number;
  search?: string;
  role?: string;
}) => {
  const { page = 1, limit = 20, search, role } = params ?? {};

  const where = {
    ...(search && {
      OR: [
        { name: { contains: search, mode: 'insensitive' as const } },
        { email: { contains: search, mode: 'insensitive' as const } },
      ],
    }),
    ...(role && { role }),
  };

  const [users, total] = await Promise.all([
    prisma.user.findMany({
      where,
      skip: (page - 1) * limit,
      take: limit,
      orderBy: { createdAt: 'desc' },
      select: {
        id: true,
        name: true,
        email: true,
        role: true,
        avatarUrl: true,
        createdAt: true,
      },
    }),
    prisma.user.count({ where }),
  ]);

  return {
    users,
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit),
    },
  };
});

export const getUser = cache(async (id: string) => {
  const user = await prisma.user.findUnique({
    where: { id },
    include: {
      orders: {
        orderBy: { createdAt: 'desc' },
        take: 10,
      },
      _count: {
        select: { orders: true },
      },
    },
  });

  if (!user) {
    throw new Error(`User not found: ${id}`);
  }

  return user;
});

// features/users/api/actions.ts
// Server Actions（フォームやクライアントからの mutation）
'use server';

import { revalidatePath } from 'next/cache';
import { prisma } from '@/shared/lib/db';
import { z } from 'zod';
import { CreateUserInput, UpdateUserInput } from '../types/user';

const createUserSchema = z.object({
  name: z.string().min(1, 'Name is required').max(100),
  email: z.string().email('Invalid email address'),
  role: z.enum(['admin', 'member', 'viewer']).default('member'),
});

export async function createUser(input: CreateUserInput) {
  const validated = createUserSchema.parse(input);

  const existingUser = await prisma.user.findUnique({
    where: { email: validated.email },
  });

  if (existingUser) {
    return { error: 'A user with this email already exists' };
  }

  const user = await prisma.user.create({
    data: validated,
  });

  revalidatePath('/dashboard');
  revalidatePath('/users');

  return { data: user };
}

export async function updateUser(id: string, input: UpdateUserInput) {
  const user = await prisma.user.update({
    where: { id },
    data: input,
  });

  revalidatePath('/dashboard');
  revalidatePath(`/users/${id}`);
  revalidatePath('/users');

  return { data: user };
}

export async function deleteUser(id: string) {
  await prisma.user.delete({ where: { id } });

  revalidatePath('/dashboard');
  revalidatePath('/users');

  return { success: true };
}
```

### 3.3 型定義の設計

```typescript
// features/users/types/user.ts
// feature 内で使用する型定義

export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  avatarUrl: string | null;
  createdAt: Date;
  updatedAt: Date;
}

export type UserRole = 'admin' | 'member' | 'viewer';

// API リクエスト用の型
export interface CreateUserInput {
  name: string;
  email: string;
  role?: UserRole;
}

export interface UpdateUserInput {
  name?: string;
  email?: string;
  role?: UserRole;
  avatarUrl?: string | null;
}

// ページネーション付きレスポンス
export interface UsersResponse {
  users: User[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

// フィルター条件
export interface UserFilters {
  search?: string;
  role?: UserRole;
  page?: number;
  limit?: number;
}

// ユーザー詳細（リレーション含む）
export interface UserDetail extends User {
  orders: {
    id: string;
    total: number;
    status: string;
    createdAt: Date;
  }[];
  _count: {
    orders: number;
  };
}
```

### 3.4 依存ルールの実装

Feature 間の依存関係を厳格に管理することが、Feature-based 構成の最も重要な要素である。

```
依存ルール（Dependency Rules）:

  許可される依存:
  ✓ features/users/  → shared/         （共有リソースの利用）
  ✓ app/            → features/users/   （ページからの利用）
  ✓ app/            → shared/           （共有リソースの利用）

  禁止される依存:
  ✗ features/users/  → features/orders/ （feature間の直接参照）
  ✗ shared/          → features/users/  （共有から個別featureへの参照）
  ✗ features/users/内部 → 外部から直接（index.ts経由でのみ）

  feature間の連携が必要な場合:
  方法1: shared/ にインターフェースを定義
    shared/types/events.ts に UserEvent 型を定義
    features/users/ が UserEvent を発行
    features/notifications/ が UserEvent を購読

  方法2: app/ でオーケストレーション
    app/(app)/dashboard/page.tsx で両方のfeatureを組み合わせる

  方法3: イベントバスパターン
    shared/lib/event-bus.ts にイベントバスを定義
    各 feature が独立してイベントを発行・購読
```

```typescript
// shared/lib/event-bus.ts — feature間連携のためのイベントバス
type EventHandler<T = unknown> = (data: T) => void;

class EventBus {
  private handlers = new Map<string, Set<EventHandler>>();

  on<T>(event: string, handler: EventHandler<T>): () => void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler as EventHandler);

    // unsubscribe 関数を返す
    return () => {
      this.handlers.get(event)?.delete(handler as EventHandler);
    };
  }

  emit<T>(event: string, data: T): void {
    this.handlers.get(event)?.forEach(handler => handler(data));
  }
}

export const eventBus = new EventBus();

// shared/types/events.ts
export interface UserCreatedEvent {
  userId: string;
  email: string;
  name: string;
}

export interface OrderCompletedEvent {
  orderId: string;
  userId: string;
  total: number;
}

// features/users/api/actions.ts — イベント発行
import { eventBus } from '@/shared/lib/event-bus';
import type { UserCreatedEvent } from '@/shared/types/events';

export async function createUser(input: CreateUserInput) {
  const user = await prisma.user.create({ data: input });

  eventBus.emit<UserCreatedEvent>('user:created', {
    userId: user.id,
    email: user.email,
    name: user.name,
  });

  return { data: user };
}

// features/notifications/hooks/useUserEvents.ts — イベント購読
import { useEffect } from 'react';
import { eventBus } from '@/shared/lib/event-bus';
import type { UserCreatedEvent } from '@/shared/types/events';

export function useUserEvents() {
  useEffect(() => {
    const unsubscribe = eventBus.on<UserCreatedEvent>(
      'user:created',
      (data) => {
        // 通知を表示
        showNotification(`New user: ${data.name}`);
      }
    );

    return unsubscribe;
  }, []);
}
```

---

## 4. パスエイリアスと Import 管理

### 4.1 TypeScript パスエイリアスの設定

パスエイリアスを設定することで、相対パスの複雑さを解消し、コードの可読性と保守性を大幅に向上させる。

```json
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"],
      "@features/*": ["./src/features/*"],
      "@shared/*": ["./src/shared/*"],
      "@app/*": ["./src/app/*"],
      "@tests/*": ["./tests/*"]
    }
  }
}
```

```typescript
// パスエイリアスの使用例

// BAD: 相対パスはネストが深いと読みにくい
import { Button } from '../../../shared/components/ui/Button';
import { useUsers } from '../../users/hooks/useUsers';
import { formatDate } from '../../../shared/lib/utils';

// GOOD: パスエイリアスで明確
import { Button } from '@/shared/components/ui/Button';
import { useUsers } from '@/features/users';
import { formatDate } from '@/shared/lib/utils';

// さらに良い: feature の index.ts 経由
import { UserList, useUsers, type User } from '@features/users';
import { Button, Modal, Table } from '@shared/components/ui';
```

### 4.2 ESLint による Import ルールの自動化

依存ルールをドキュメントに書くだけでは不十分である。ESLint を使って自動的にルール違反を検出する。

```javascript
// .eslintrc.js
module.exports = {
  plugins: ['import', 'boundaries'],
  settings: {
    'import/resolver': {
      typescript: {
        alwaysTryTypes: true,
      },
    },
    // boundaries プラグインの設定
    'boundaries/elements': [
      { type: 'app', pattern: 'src/app/*' },
      { type: 'features', pattern: 'src/features/*' },
      { type: 'shared', pattern: 'src/shared/*' },
    ],
    'boundaries/ignore': ['**/*.test.*', '**/*.spec.*'],
  },
  rules: {
    // features 間の直接 import を禁止
    'import/no-restricted-paths': [
      'error',
      {
        zones: [
          // features/users/ から features/orders/ への import を禁止
          {
            target: './src/features/users/**',
            from: './src/features/orders/**',
            message: 'Feature modules cannot import from other features directly. Use shared/ instead.',
          },
          {
            target: './src/features/orders/**',
            from: './src/features/users/**',
            message: 'Feature modules cannot import from other features directly. Use shared/ instead.',
          },
          // shared/ から features/ への import を禁止
          {
            target: './src/shared/**',
            from: './src/features/**',
            message: 'Shared modules cannot depend on feature modules.',
          },
        ],
      },
    ],

    // import の順序を統一
    'import/order': [
      'error',
      {
        groups: [
          'builtin',       // Node.js 組み込みモジュール
          'external',      // npm パッケージ
          'internal',      // パスエイリアス
          'parent',        // 親ディレクトリ
          'sibling',       // 同階層
          'index',         // index ファイル
          'type',          // 型 import
        ],
        pathGroups: [
          { pattern: 'react', group: 'builtin', position: 'before' },
          { pattern: 'next/**', group: 'builtin', position: 'before' },
          { pattern: '@/features/**', group: 'internal', position: 'before' },
          { pattern: '@/shared/**', group: 'internal', position: 'after' },
        ],
        'newlines-between': 'always',
        alphabetize: { order: 'asc', caseInsensitive: true },
      },
    ],

    // feature 内部ファイルへの直接アクセスを禁止
    'no-restricted-imports': [
      'error',
      {
        patterns: [
          {
            group: ['@/features/*/components/*', '@/features/*/hooks/*', '@/features/*/api/*'],
            message: 'Import from the feature index file instead: @/features/<name>',
          },
        ],
      },
    ],
  },
};
```

### 4.3 Barrel Export パターンとその注意点

```typescript
// Barrel Export（index.ts からの再 export）
// src/shared/components/ui/index.ts
export { Button } from './Button';
export { Input } from './Input';
export { Modal } from './Modal';
export { Table } from './Table';
export { Select } from './Select';
export { Checkbox } from './Checkbox';
export { Skeleton } from './Skeleton';
export { Badge } from './Badge';
export { Card, CardHeader, CardContent, CardFooter } from './Card';
export { Tabs, TabsList, TabsTrigger, TabsContent } from './Tabs';

// 使用側
import { Button, Modal, Table } from '@/shared/components/ui';
```

**Barrel Export の注意点:**

```
利点:
  ✓ import 文が短くなり、可読性が向上する
  ✓ 公開 API を明示的に制御できる
  ✓ 内部実装の変更が外部に影響しない

注意点:
  ✗ Tree-shaking の阻害
    バンドラーが未使用の export を除去できない場合がある
    特に Server Components では影響が大きい

  ✗ 循環参照のリスク
    複数の barrel file が互いを参照すると循環が発生する

  ✗ パフォーマンス
    大量の re-export はモジュール解決のオーバーヘッドになる

対策:
  - features/ の index.ts は使い分ける（推奨）
  - shared/components/ui/ は barrel OK
  - Next.js では modularizeImports の設定を活用
```

```javascript
// next.config.js — modularizeImports で barrel export の問題を回避
/** @type {import('next').NextConfig} */
const nextConfig = {
  modularizeImports: {
    // lodash の個別インポートに自動変換
    'lodash': {
      transform: 'lodash/{{member}}',
    },
    // @/shared/components/ui の個別インポートに自動変換
    '@/shared/components/ui': {
      transform: '@/shared/components/ui/{{member}}',
    },
  },
};

module.exports = nextConfig;
```

---

## 5. モノレポ設計

### 5.1 モノレポの基本構成

複数のアプリケーションやパッケージを1つのリポジトリで管理するモノレポは、中〜大規模プロジェクトで採用される。

```
モノレポ構成（Turborepo + pnpm）:

  monorepo/
  ├── apps/                     ← アプリケーション
  │   ├── web/                  ← Next.js フロントエンド
  │   │   ├── src/
  │   │   │   ├── app/
  │   │   │   ├── features/
  │   │   │   └── shared/
  │   │   ├── next.config.js
  │   │   ├── tailwind.config.ts
  │   │   ├── tsconfig.json     ← extends: @repo/typescript-config
  │   │   └── package.json
  │   ├── admin/                ← 管理画面
  │   │   ├── src/
  │   │   ├── next.config.js
  │   │   └── package.json
  │   ├── api/                  ← バックエンドAPI（Hono / Express）
  │   │   ├── src/
  │   │   │   ├── routes/
  │   │   │   ├── services/
  │   │   │   └── middleware/
  │   │   └── package.json
  │   ├── docs/                 ← ドキュメントサイト
  │   │   └── package.json
  │   └── mobile/               ← React Native / Expo
  │       └── package.json
  ├── packages/                 ← 共有パッケージ
  │   ├── ui/                   ← 共有UIコンポーネント
  │   │   ├── src/
  │   │   │   ├── Button.tsx
  │   │   │   ├── Modal.tsx
  │   │   │   └── index.ts
  │   │   ├── tsconfig.json
  │   │   └── package.json
  │   ├── db/                   ← Prismaスキーマ + クライアント
  │   │   ├── prisma/
  │   │   │   └── schema.prisma
  │   │   ├── src/
  │   │   │   └── client.ts
  │   │   └── package.json
  │   ├── auth/                 ← 認証ロジック
  │   │   ├── src/
  │   │   └── package.json
  │   ├── email/                ← メールテンプレート
  │   │   ├── src/
  │   │   └── package.json
  │   ├── config/               ← 共有設定
  │   │   ├── eslint/
  │   │   │   ├── base.js
  │   │   │   ├── next.js
  │   │   │   └── package.json
  │   │   └── typescript/
  │   │       ├── base.json
  │   │       ├── next.json
  │   │       ├── node.json
  │   │       └── package.json
  │   ├── types/                ← 共有型定義
  │   │   ├── src/
  │   │   │   ├── user.ts
  │   │   │   ├── order.ts
  │   │   │   └── index.ts
  │   │   └── package.json
  │   └── utils/                ← 共有ユーティリティ
  │       ├── src/
  │       │   ├── format.ts
  │       │   ├── validation.ts
  │       │   └── index.ts
  │       └── package.json
  ├── tooling/                  ← 開発ツール設定
  │   ├── github/
  │   │   └── workflows/
  │   └── docker/
  │       ├── Dockerfile.web
  │       └── docker-compose.yml
  ├── turbo.json                ← Turborepo 設定
  ├── pnpm-workspace.yaml       ← pnpm ワークスペース定義
  ├── package.json              ← ルート package.json
  ├── .gitignore
  └── .env.example
```

### 5.2 Turborepo の設定

```json
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": [".next/**", "!.next/cache/**", "dist/**"],
      "env": ["DATABASE_URL", "NEXT_PUBLIC_*"]
    },
    "dev": {
      "dependsOn": ["^build"],
      "cache": false,
      "persistent": true
    },
    "lint": {
      "dependsOn": ["^build"]
    },
    "test": {
      "dependsOn": ["^build"],
      "outputs": ["coverage/**"]
    },
    "test:e2e": {
      "dependsOn": ["^build"],
      "outputs": [],
      "cache": false
    },
    "type-check": {
      "dependsOn": ["^build"]
    },
    "clean": {
      "cache": false
    }
  }
}
```

```yaml
# pnpm-workspace.yaml
packages:
  - "apps/*"
  - "packages/*"
  - "tooling/*"
```

```json
// package.json（ルート）
{
  "name": "my-monorepo",
  "private": true,
  "scripts": {
    "dev": "turbo dev",
    "build": "turbo build",
    "lint": "turbo lint",
    "test": "turbo test",
    "type-check": "turbo type-check",
    "clean": "turbo clean",
    "format": "prettier --write \"**/*.{ts,tsx,js,jsx,json,md}\"",
    "db:push": "pnpm --filter @repo/db db:push",
    "db:studio": "pnpm --filter @repo/db db:studio",
    "db:generate": "pnpm --filter @repo/db db:generate"
  },
  "devDependencies": {
    "prettier": "^3.2.0",
    "turbo": "^2.0.0"
  },
  "packageManager": "pnpm@9.0.0"
}
```

### 5.3 共有パッケージの作成

```json
// packages/ui/package.json
{
  "name": "@repo/ui",
  "version": "0.0.0",
  "private": true,
  "main": "./src/index.ts",
  "types": "./src/index.ts",
  "exports": {
    ".": "./src/index.ts",
    "./*": "./src/*.tsx"
  },
  "dependencies": {
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0"
  },
  "peerDependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "devDependencies": {
    "@repo/typescript-config": "workspace:*",
    "typescript": "^5.3.0"
  }
}
```

```typescript
// packages/ui/src/Button.tsx
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from './utils';
import { forwardRef } from 'react';

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground hover:bg-primary/90',
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        outline: 'border border-input bg-background hover:bg-accent',
        secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        link: 'text-primary underline-offset-4 hover:underline',
      },
      size: {
        default: 'h-10 px-4 py-2',
        sm: 'h-9 rounded-md px-3',
        lg: 'h-11 rounded-md px-8',
        icon: 'h-10 w-10',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = 'Button';

export { Button, buttonVariants };

// packages/ui/src/index.ts
export { Button, type ButtonProps, buttonVariants } from './Button';
export { Input, type InputProps } from './Input';
export { Modal, type ModalProps } from './Modal';
export { Table } from './Table';
```

```typescript
// apps/web/src/app/page.tsx — 共有パッケージの使用
import { Button } from '@repo/ui';
import { formatDate } from '@repo/utils';
import type { User } from '@repo/types';

export default function HomePage() {
  return (
    <div>
      <h1>Welcome</h1>
      <Button variant="default" size="lg">
        Get Started
      </Button>
    </div>
  );
}
```

### 5.4 モノレポツールの比較

```
ツール詳細比較:

  特徴              | Turborepo      | Nx              | pnpm workspace
  ─────────────────|───────────────|────────────────|──────────────────
  リモートキャッシュ  | ✓ (Vercel)    | ✓ (Nx Cloud)   | ✗
  ローカルキャッシュ  | ✓             | ✓              | ✗
  タスク並列実行     | ✓             | ✓              | 限定的
  依存グラフ可視化   | ✓             | ✓ (充実)       | ✗
  コード生成        | ✗              | ✓ (Generator)  | ✗
  プラグイン        | 少ない          | 豊富           | ✗
  設定量           | 少ない          | 中程度          | 最小
  学習コスト        | 低い           | 中程度          | 最低
  パフォーマンス    | 高速           | 高速           | 中程度
  推奨プロジェクト   | Web/フロント   | エンタープライズ  | 小規模

  選択ガイドライン:
  - Web フロントエンド中心 → Turborepo（Vercel との統合が強力）
  - エンタープライズ/大規模 → Nx（ツール・プラグインが豊富）
  - 最小構成でシンプルに → pnpm workspace（追加ツール不要）
  - React Native 含む → Turborepo or Nx
```

### 5.5 モノレポのCI/CD設計

```yaml
# .github/workflows/ci.yml
name: CI
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

env:
  TURBO_TOKEN: ${{ secrets.TURBO_TOKEN }}
  TURBO_TEAM: ${{ vars.TURBO_TEAM }}

jobs:
  lint-and-type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v3
        with:
          version: 9
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'pnpm'
      - run: pnpm install --frozen-lockfile
      - run: pnpm lint
      - run: pnpm type-check

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v3
        with:
          version: 9
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'pnpm'
      - run: pnpm install --frozen-lockfile
      - run: pnpm test

  build:
    runs-on: ubuntu-latest
    needs: [lint-and-type-check, test]
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v3
        with:
          version: 9
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'pnpm'
      - run: pnpm install --frozen-lockfile
      - run: pnpm build
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}

  # 影響を受けたパッケージのみデプロイ
  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2  # turbo の変更検出に必要
      - uses: pnpm/action-setup@v3
        with:
          version: 9
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'pnpm'
      - run: pnpm install --frozen-lockfile
      # 変更があった app のみをデプロイ
      - run: pnpm turbo build --filter=...[HEAD~1]
```

---

## 6. テスト配置戦略

### 6.1 テストファイルの配置パターン

テストファイルの配置には複数のパターンがあり、プロジェクトの特性に応じて選択する。

```
パターン1: コロケーション（推奨）
  テストを対象ファイルの近くに配置する

  features/users/
  ├── components/
  │   ├── UserList.tsx
  │   ├── UserList.test.tsx      ← コンポーネントテスト
  │   ├── UserCard.tsx
  │   └── UserCard.test.tsx
  ├── hooks/
  │   ├── useUsers.ts
  │   └── useUsers.test.ts       ← hooks テスト
  ├── api/
  │   ├── actions.ts
  │   └── actions.test.ts        ← Server Actions テスト
  └── utils/
      ├── format.ts
      └── format.test.ts         ← ユーティリティテスト

  利点:
  ✓ テスト対象との関連が一目瞭然
  ✓ feature を削除するとテストも一緒に消える
  ✓ テストの追加忘れに気づきやすい

パターン2: __tests__ ディレクトリ
  feature 内に __tests__ ディレクトリを作成

  features/users/
  ├── components/
  │   ├── UserList.tsx
  │   └── UserCard.tsx
  ├── hooks/
  │   └── useUsers.ts
  └── __tests__/
      ├── UserList.test.tsx
      ├── UserCard.test.tsx
      └── useUsers.test.ts

  利点:
  ✓ テストとプロダクションコードが明確に分離
  ✓ ファイルツリーがすっきりする
  ✓ テストのみを対象にした操作がしやすい

パターン3: トップレベル tests ディレクトリ
  統合テストとE2Eテストに使用

  project-root/
  ├── src/
  │   └── features/
  └── tests/
      ├── unit/                  ← ユニットテスト（非推奨、コロケーション推奨）
      ├── integration/           ← 統合テスト
      │   ├── users.test.ts
      │   └── orders.test.ts
      └── e2e/                   ← E2Eテスト（Playwright）
          ├── auth.spec.ts
          ├── dashboard.spec.ts
          └── fixtures/
              └── test-data.ts
```

### 6.2 テスト設定ファイル

```typescript
// vitest.config.ts — ユニットテスト設定
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.ts'],
    include: [
      'src/**/*.test.{ts,tsx}',
      'src/**/*.spec.{ts,tsx}',
    ],
    exclude: [
      'node_modules',
      'tests/e2e/**',
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.{ts,tsx}'],
      exclude: [
        'src/**/*.test.{ts,tsx}',
        'src/**/*.d.ts',
        'src/**/index.ts',    // barrel files
        'src/app/**',          // ルーティングファイル
      ],
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@features': path.resolve(__dirname, './src/features'),
      '@shared': path.resolve(__dirname, './src/shared'),
    },
  },
});

// playwright.config.ts — E2Eテスト設定
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? 'github' : 'html',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'mobile-chrome', use: { ...devices['Pixel 5'] } },
  ],
  webServer: {
    command: 'pnpm dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});
```

### 6.3 テストの実践例

```typescript
// features/users/components/UserList.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { UserList } from './UserList';

// Server Component のテスト（async component）
describe('UserList', () => {
  it('ユーザー一覧を表示する', async () => {
    // Server Component は直接レンダリングをテスト
    const result = await UserList({ page: 1 });
    render(result);

    expect(screen.getByRole('table')).toBeInTheDocument();
    expect(screen.getAllByRole('row')).toHaveLength(21); // header + 20 rows
  });

  it('ユーザーが0件の場合、空状態メッセージを表示する', async () => {
    vi.mocked(getUsers).mockResolvedValueOnce({
      users: [],
      pagination: { page: 1, limit: 20, total: 0, totalPages: 0 },
    });

    const result = await UserList({ page: 1 });
    render(result);

    expect(screen.getByText('No users found')).toBeInTheDocument();
  });
});

// features/users/hooks/useUsers.test.ts
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, it, expect, vi } from 'vitest';
import { useUsers } from './useUsers';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}

describe('useUsers', () => {
  it('ユーザー一覧を取得できる', async () => {
    const { result } = renderHook(() => useUsers(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data?.users).toHaveLength(20);
  });

  it('検索フィルターを適用できる', async () => {
    const { result } = renderHook(
      () => useUsers({ search: 'John' }),
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data?.users).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: expect.stringContaining('John') }),
      ])
    );
  });
});

// features/users/api/actions.test.ts — Server Actions のテスト
import { describe, it, expect, beforeEach } from 'vitest';
import { createUser, updateUser, deleteUser } from './actions';
import { prisma } from '@/shared/lib/db';

// Prisma のモック
vi.mock('@/shared/lib/db', () => ({
  prisma: {
    user: {
      create: vi.fn(),
      findUnique: vi.fn(),
      update: vi.fn(),
      delete: vi.fn(),
    },
  },
}));

describe('createUser', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('有効な入力でユーザーを作成できる', async () => {
    vi.mocked(prisma.user.findUnique).mockResolvedValue(null);
    vi.mocked(prisma.user.create).mockResolvedValue({
      id: '1',
      name: 'John Doe',
      email: 'john@example.com',
      role: 'member',
      avatarUrl: null,
      createdAt: new Date(),
      updatedAt: new Date(),
    });

    const result = await createUser({
      name: 'John Doe',
      email: 'john@example.com',
    });

    expect(result.data).toBeDefined();
    expect(result.data?.name).toBe('John Doe');
    expect(prisma.user.create).toHaveBeenCalledOnce();
  });

  it('既存メールアドレスの場合エラーを返す', async () => {
    vi.mocked(prisma.user.findUnique).mockResolvedValue({
      id: '1',
      name: 'Existing',
      email: 'john@example.com',
      role: 'member',
      avatarUrl: null,
      createdAt: new Date(),
      updatedAt: new Date(),
    });

    const result = await createUser({
      name: 'John Doe',
      email: 'john@example.com',
    });

    expect(result.error).toBe('A user with this email already exists');
    expect(prisma.user.create).not.toHaveBeenCalled();
  });

  it('バリデーション違反の場合エラーをスローする', async () => {
    await expect(
      createUser({ name: '', email: 'invalid' })
    ).rejects.toThrow();
  });
});
```

### 6.4 テストのファイル命名規則

```
命名規則の比較:

  パターン              | 例                      | 用途
  ────────────────────|──────────────────────────|──────────────
  *.test.ts(x)        | UserList.test.tsx        | ユニットテスト
  *.spec.ts(x)        | auth.spec.ts             | E2Eテスト
  *.integration.ts    | api.integration.ts       | 統合テスト
  *.stories.tsx       | Button.stories.tsx       | Storybook

  推奨:
  - ユニットテスト → *.test.ts(x)（コロケーション）
  - E2Eテスト → *.spec.ts（tests/e2e/ ディレクトリ）
  - 統合テスト → *.integration.ts（tests/integration/）
  - Storybook → *.stories.tsx（コロケーション）
```

---

## 7. 命名規則とファイル構成ガイドライン

### 7.1 ファイル命名規則

```
ファイル命名規則一覧:

  種類                    | 命名規則        | 例
  ──────────────────────|───────────────|────────────────────
  React コンポーネント     | PascalCase    | UserList.tsx
  カスタム hooks          | camelCase     | useUsers.ts
  ユーティリティ関数       | camelCase     | formatDate.ts
  型定義ファイル           | camelCase     | user.ts
  定数ファイル            | camelCase     | routes.ts
  テストファイル           | 対象名.test   | UserList.test.tsx
  Storybook              | 対象名.stories | Button.stories.tsx
  Server Actions         | camelCase     | actions.ts
  API クエリ              | camelCase     | queries.ts
  設定ファイル            | kebab-case    | next.config.js
  CSS/スタイル            | kebab-case    | globals.css
  ディレクトリ名          | kebab-case    | user-profile/
  feature ディレクトリ    | kebab-case    | order-management/

  特殊なNext.jsファイル:
  - page.tsx, layout.tsx, loading.tsx, error.tsx, not-found.tsx
  - route.ts (API Route Handlers)
  - middleware.ts
  - template.tsx (レイアウトのリセット版)
  - default.tsx (Parallel Routes のデフォルト)
```

### 7.2 ディレクトリ命名規則

```typescript
// ディレクトリ名は kebab-case を使用
// GOOD
src/features/user-management/
src/features/order-processing/
src/shared/components/data-table/

// BAD
src/features/UserManagement/    // PascalCase はコンポーネント名に予約
src/features/userManagement/    // camelCase はファイル名に予約
src/features/user_management/   // snake_case は使わない

// 例外: Next.js の動的ルートとルートグループ
src/app/[slug]/                 // 動的セグメント
src/app/(marketing)/            // ルートグループ
src/app/[...catchAll]/          // キャッチオールセグメント
src/app/@modal/                 // Parallel Routes（Named Slot）
src/app/(.)photo/               // Intercepting Routes
```

### 7.3 Export の命名規則

```typescript
// コンポーネント: PascalCase の名前付き export
// features/users/components/UserList.tsx
export function UserList({ users }: UserListProps) { ... }

// BAD: default export（名前の一貫性が保てない）
export default function UserList() { ... }
// 別ファイルで import するとき任意の名前にできてしまう:
// import UsersList from './UserList';  // 異なる名前でも動く

// hooks: camelCase の名前付き export
// features/users/hooks/useUsers.ts
export function useUsers(filters?: UserFilters) { ... }

// 型: PascalCase の名前付き export（type キーワード付き）
// features/users/types/user.ts
export interface User { ... }
export type UserRole = 'admin' | 'member' | 'viewer';

// 定数: UPPER_SNAKE_CASE
// shared/constants/config.ts
export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
export const SUPPORTED_FORMATS = ['image/jpeg', 'image/png', 'image/webp'] as const;
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:3001';

// enum の代替: as const オブジェクト
export const USER_ROLES = {
  ADMIN: 'admin',
  MEMBER: 'member',
  VIEWER: 'viewer',
} as const;

export type UserRole = typeof USER_ROLES[keyof typeof USER_ROLES];
```

---

## 8. 設計チェックリスト

### 8.1 プロジェクト初期設定チェックリスト

```
プロジェクト初期設定チェックリスト:

  ディレクトリ構成:
  □ Feature-based でモジュール分割されているか
  □ 各 feature が index.ts で公開APIを定義しているか
  □ features 間の直接 import が禁止されているか
  □ 共有コンポーネントが shared/ にあるか
  □ app/ にはルーティングのみか
  □ テストファイルの配置ルールが決まっているか

  命名規則:
  □ コンポーネント → PascalCase
  □ hooks → camelCase (use プレフィックス)
  □ ディレクトリ → kebab-case
  □ 定数 → UPPER_SNAKE_CASE
  □ パスエイリアスが設定されているか (@/, @features/, @shared/)
  □ テストファイルが対象の近くにあるか

  ツール設定:
  □ TypeScript の strict mode が有効か
  □ ESLint の import ルールが設定されているか
  □ Prettier が設定されているか
  □ husky + lint-staged で pre-commit チェックがあるか
  □ CI/CD パイプラインが設定されているか

  スケーラビリティ:
  □ 新機能追加が既存コードに影響しないか
  □ feature の削除が容易か
  □ チームメンバーが迷わず配置できるか
  □ テストが feature 単位で実行できるか

  ドキュメント:
  □ ARCHITECTURE.md が存在するか
  □ feature の追加手順が文書化されているか
  □ 依存ルールが明文化されているか
  □ ADR（Architecture Decision Records）が整備されているか
```

### 8.2 コードレビューチェックリスト

```
コードレビュー時のプロジェクト構成チェック:

  ファイル配置:
  □ 新しいファイルは正しいディレクトリに配置されているか
  □ feature 固有のコードが shared/ に配置されていないか
  □ 汎用コードが feature/ 内に閉じ込められていないか
  □ テストファイルが追加されているか

  依存関係:
  □ features 間の直接 import がないか
  □ shared/ から features/ への依存がないか
  □ 循環参照が発生していないか
  □ 不要な依存関係が追加されていないか

  公開API:
  □ 新しいコンポーネント/hooks が index.ts に追加されているか
  □ 内部実装が不必要に公開されていないか
  □ 型が適切に export されているか

  命名:
  □ 命名規則に従っているか
  □ コンポーネント名がファイル名と一致しているか
  □ 汎用的すぎる命名（data, info, handler）を避けているか
```

### 8.3 ARCHITECTURE.md テンプレート

```markdown
# Architecture

## ディレクトリ構成

このプロジェクトは Feature-based 構成を採用しています。

### ルール
1. `app/` にはルーティングとページコンポーネントのみ配置
2. ビジネスロジックは `features/` に配置
3. 2つ以上の feature で使うものは `shared/` に配置
4. features 間の直接 import は禁止

### 新しい feature の追加手順
1. `src/features/<feature-name>/` ディレクトリを作成
2. components/, hooks/, api/, types/ サブディレクトリを作成
3. index.ts で公開APIを定義
4. テストを追加
5. ESLint ルールに新しい feature を追加

### 依存関係図
app/ → features/ → shared/
app/ → shared/

### ADR
決定記録は docs/adr/ に保管しています。
```

---

## 9. トラブルシューティング

### 9.1 よくある問題と解決策

```
問題1: 「この機能はどの feature に属するか判断できない」

  解決策:
  - ドメインエキスパートに確認する
  - ユーザーストーリーやユースケースに基づいて判断する
  - 迷ったら仮配置して、後でリファクタリングする
  - 2つの feature にまたがる場合は、新しい feature を作るか shared/ に配置

問題2: 「shared/ が肥大化して型ベース構成に逆戻りしている」

  解決策:
  - shared/ にも構造を持たせる（components/ui/, hooks/, lib/）
  - 本当に2つ以上の feature で使われているか定期的に確認
  - 1つの feature でしか使われていないものは feature 内に移動

問題3: 「feature 間で共有したいロジックがある」

  解決策:
  - shared/lib/ にユーティリティとして切り出す
  - shared/types/ に共通の型を定義する
  - イベントバスパターンで疎結合な連携を実現する
  - React Context を shared/ に配置する

問題4: 「循環参照が発生している」

  解決策:
  - madge などのツールで依存グラフを可視化する
  - 共通の依存を shared/ に抽出する
  - インターフェース（型）を shared/ に定義して依存を逆転させる
  - ESLint の import/no-cycle ルールを有効にする

問題5: 「テストが壊れやすい（Fragile Tests）」

  解決策:
  - 内部実装ではなく公開APIに対してテストする
  - index.ts の export のみをテスト対象にする
  - モックは最小限にする
  - テストの独立性を確保する（テスト間の依存を排除）
```

### 9.2 パフォーマンス最適化

```typescript
// 問題: barrel export による不要なモジュール読み込み
// BAD: shared/components/ui/index.ts から1つだけ使う場合でも
// 全コンポーネントが読み込まれる可能性がある
import { Button } from '@/shared/components/ui';

// GOOD: 直接インポート（Tree-shaking に確実に効く）
import { Button } from '@/shared/components/ui/Button';

// GOOD: next.config.js で modularizeImports を設定（前述）

// 問題: 大きな feature モジュールの動的インポート
// クライアントサイドで重い feature を遅延読み込み
import dynamic from 'next/dynamic';

const HeavyChart = dynamic(
  () => import('@/features/analytics/components/HeavyChart'),
  {
    loading: () => <Skeleton className="h-64" />,
    ssr: false, // クライアントサイドのみ
  }
);

// 問題: Server Components でのデータプリフェッチ
// features/users/components/UserList.tsx
import { Suspense } from 'react';
import { getUsers } from '../api/queries';

// Streaming SSR でデータ取得を並列化
export async function UserList() {
  const { users } = await getUsers();

  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

---

## FAQ

### Q1: モノレポとマルチレポ、どちらを選ぶべきか？

**A:** プロジェクト規模とチーム構成で判断します。

**モノレポが適している場合:**

| メリット | 説明 |
|---------|------|
| コード共有が容易 | shared/ パッケージで型やユーティリティを一元管理 |
| 依存関係の一貫性 | 全パッケージでライブラリバージョンを統一可能 |
| アトミックな変更 | APIとフロントエンドを同一PRで変更できる |
| リファクタリング効率 | IDE の Rename Symbol で全パッケージを一括修正 |

例: マイクロサービス（Web/Mobile/Admin）、複数プロダクト共通基盤

**マルチレポが適している場合:**

| メリット | 説明 |
|---------|------|
| 独立したリリースサイクル | 各リポジトリが個別にデプロイ可能 |
| アクセス制御 | リポジトリ単位で権限を分離 |
| ビルド速度 | 影響範囲が限定され、CI/CDが高速 |

例: 完全独立したプロダクト群、異なるチームが管理

**ハイブリッドアプローチ:**
- メインアプリ → モノレポ（apps/web, apps/mobile, packages/shared）
- 独立プロダクト → 別リポジトリ
- 共通ライブラリ → npm パッケージとして公開

### Q2: Feature-based構成とLayer-based構成、どちらが優れているか？

**A:** 規模と要件に応じて使い分けます。

**Feature-based構成の強み:**

```
features/
├── user-management/     ← 機能単位で完結
│   ├── components/      ← この機能専用のUI
│   ├── hooks/           ← この機能専用のロジック
│   ├── api/             ← この機能専用のAPI通信
│   └── types/           ← この機能専用の型定義
└── order-processing/    ← 別の機能も同様
```

- **凝集度が高い**: 関連するコードが物理的に近接
- **削除が容易**: 機能を廃止する時はディレクトリごと削除
- **チームスケール**: 複数チームが独立して作業可能
- **適用場面**: 中〜大規模アプリ（10+ 機能）

**Layer-based構成の強み:**

```
src/
├── components/    ← すべてのUIコンポーネント
├── hooks/         ← すべてのカスタムフック
├── api/           ← すべてのAPI通信
└── types/         ← すべての型定義
```

- **シンプル**: 技術的な役割で明確に分類
- **学習コスト低**: 初心者でも理解しやすい
- **適用場面**: 小規模アプリ（5機能未満）、プロトタイプ

**推奨戦略:**
1. プロジェクト開始時は Layer-based でスタート
2. 機能が5つを超えたタイミングで Feature-based に移行
3. shared/ には共通コンポーネント（Button, Modal等）を配置

### Q3: プロジェクト構成を変更するタイミングは？

**A:** 以下のシグナルが見えたら構成見直しを検討します。

**構成変更が必要なシグナル:**

| シグナル | 症状 | 対処 |
|---------|------|------|
| ファイル発見に時間がかかる | 「このコードはどこ？」と毎回検索 | Feature-based に移行 |
| import 文が長大化 | `../../../components/UserCard` | パスエイリアス導入 |
| 循環参照が頻発 | 機能間の依存が複雑化 | 依存グラフ可視化 → リファクタリング |
| 同じコードの重複 | 複数箇所で同じロジック | shared/ に共通化 |
| テストが壊れやすい | 内部実装変更でテストが失敗 | 公開API（index.ts）のみをテスト |
| ビルド時間が長い | 5分以上かかる | Turborepo導入、モノレポ最適化 |

**段階的な移行手順:**

```bash
# Step 1: パスエイリアス導入（影響範囲: 低）
# tsconfig.json に paths 設定を追加
# 相対パス → @/features/xxx に置換

# Step 2: shared/ ディレクトリ作成（影響範囲: 中）
# 共通コンポーネントを shared/components/ に移動
# 共通hooks を shared/hooks/ に移動

# Step 3: features/ ディレクトリ作成（影響範囲: 高）
# 機能ごとにディレクトリを作成
# 関連ファイルを機能ディレクトリに移動
# index.ts で公開APIを定義

# Step 4: ESLint ルール追加（影響範囲: 低）
# no-restricted-imports で依存ルールを強制

# Step 5: モノレポ化（影響範囲: 最高）
# Turborepo 導入、パッケージ分割
```

**タイミングの目安:**
- プロジェクト開始〜3ヶ月: Layer-based でOK
- 3〜6ヶ月（機能5つ以上）: Feature-based 移行を検討
- 6ヶ月以上（複数アプリ）: モノレポ化を検討

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Feature-based | 機能ごとにディレクトリを分割し、高凝集・低結合を実現 |
| 公開API | index.ts で外部向けインターフェースを定義し、内部実装を隠蔽 |
| 依存ルール | features間は直接参照禁止。ESLint で自動チェック |
| App Router | app/ はルーティングのみ、ビジネスロジックは features/ |
| モノレポ | Turborepo + pnpm workspace で複数アプリを統合管理 |
| パスエイリアス | @features/, @shared/ で可読性向上と移行容易性確保 |
| テスト配置 | コロケーション推奨。E2Eテストはトップレベル tests/ に |
| 命名規則 | コンポーネント=PascalCase, hooks=camelCase, ディレクトリ=kebab-case |
| Barrel Export | 便利だが Tree-shaking への影響に注意。modularizeImports で対策 |

---

## 次に読むべきガイド

---

## 参考文献
1. Bulletproof React. "Project Structure." github.com/alan2207/bulletproof-react, 2024.
2. Next.js. "Project Organization." nextjs.org/docs, 2024.
3. Turborepo. "Getting Started." turbo.build, 2024.
4. Nx. "Why Nx?" nx.dev, 2024.
5. Kent C. Dodds. "Colocation." kentcdodds.com, 2019.
6. Dan Abramov. "Presentational and Container Components." (Deprecated pattern, but historically important), 2015.
7. Feature-Sliced Design. "Architectural methodology for frontend projects." feature-sliced.design, 2024.
8. Mark Erikson. "Scaling React Applications." Redux documentation, 2024.
9. Vercel. "Monorepos with Turborepo." vercel.com/docs, 2024.
10. pnpm. "Workspace." pnpm.io/workspaces, 2024.

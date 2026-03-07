# デプロイ先プラットフォーム完全ガイド

> デプロイ先の選択はアプリの性能・コスト・運用に直結する。Vercel、Cloudflare、AWS、Docker、GCP、Azure、Railway、Fly.io など主要プラットフォームの特徴と選定基準を理解し、プロジェクトに最適なデプロイ戦略を選択する。

## この章で学ぶこと

- [ ] 主要なデプロイプラットフォームの比較と選定基準を理解する
- [ ] 各プラットフォームの料金体系・制限事項・スケーリング特性を把握する
- [ ] Docker 化とセルフホスティングのパターンを学ぶ
- [ ] CI/CD パイプラインとの統合方法を理解する
- [ ] マルチリージョン・エッジデプロイの設計パターンを学ぶ
- [ ] コスト最適化とパフォーマンスチューニングの実践手法を習得する
- [ ] 障害対応・ロールバック・ブルーグリーンデプロイなどの運用パターンを理解する

---

## 1. プラットフォーム総合比較

### 1.1 主要プラットフォーム一覧と特性マトリクス

```
                Vercel     Cloudflare   AWS         GCP         Azure       Docker/VPS   Railway     Fly.io
──────────────────────────────────────────────────────────────────────────────────────────────────────────
対象           Next.js     全般         全般        全般        全般        全般          全般        全般
サーバーレス    ○          ○(Workers)   ○(Lambda)   ○(Cloud Run) ○(Functions) ×           ○           △
エッジ         ○          ◎            ○           ○           △           ×            ×           ○
SSR            ○          △            ○           ○           ○           ○            ○           ○
コスト         中          安い         変動        変動        変動        インフラ次第   安い        中
スケーリング   自動        自動         自動/手動    自動        自動/手動    手動/K8s      自動        自動
カスタマイズ   低          中           高          高          高          最高          中          中
学習コスト     最低        低           高          中〜高      中〜高      中            低          低
DB統合         △          D1,KV等      RDS等       Cloud SQL   Azure DB    自由          PostgreSQL  PostgreSQL
SSL/TLS        自動        自動         ACM/手動    自動/手動   自動/手動   Let's Encrypt  自動        自動
カスタムドメイン ○          ○            ○           ○           ○           ○            ○           ○
ログ/監視      基本        基本         CloudWatch  Cloud Logging App Insights 自前構築   基本        基本
```

### 1.2 コスト比較（月間トラフィック別）

```
月間100万PV想定のコスト比較:

                     月額コスト    帯域        リクエスト上限      備考
─────────────────────────────────────────────────────────────────────
Vercel Pro           $20          1TB         無制限             チーム機能含む
Cloudflare Pages     $0〜$5       無制限      無制限             Workers併用時$5
AWS Amplify          $15〜$50     変動        変動               転送量課金
AWS ECS Fargate      $50〜$200    変動        無制限             インスタンス課金
GCP Cloud Run        $10〜$100    変動        変動               リクエスト+CPU課金
Railway              $5〜$20      変動        無制限             リソース課金
Fly.io               $10〜$50     変動        無制限             VM課金
VPS (Hetzner等)      $5〜$20      固定        無制限             自己管理
```

### 1.3 選定フローチャート

```
プロジェクト要件の確認
│
├─ Next.js プロジェクト？
│  ├─ Yes → 予算は？
│  │  ├─ 無料〜$20/月 → Vercel (Hobby / Pro)
│  │  ├─ AWS 環境必須 → AWS Amplify or ECS
│  │  └─ エッジ重視 → Cloudflare Pages + Workers
│  └─ No → フレームワークは？
│     ├─ React SPA / Vue / Svelte → Cloudflare Pages or Vercel
│     ├─ Express / Fastify API → Railway / Fly.io / ECS
│     └─ Full-stack (DB含む) → Railway / AWS / GCP
│
├─ エッジでの実行が必要？
│  ├─ Yes → Cloudflare Workers or Vercel Edge Functions
│  └─ No → 次の判断基準へ
│
├─ コンテナ化が必要？
│  ├─ Yes → AWS ECS / GCP Cloud Run / Fly.io / Docker + VPS
│  └─ No → マネージドサービスを検討
│
├─ データベースの要件は？
│  ├─ 軽量 (SQLite相当) → Cloudflare D1 / Turso
│  ├─ PostgreSQL → Railway / Supabase / AWS RDS / GCP Cloud SQL
│  ├─ NoSQL → DynamoDB / Firestore / MongoDB Atlas
│  └─ グローバル分散 → PlanetScale / CockroachDB / Cloudflare D1
│
└─ 予算制約は？
   ├─ 無料枠で運用 → Cloudflare Pages / Vercel Hobby / Railway Free
   ├─ $20以下 → Vercel Pro / Railway / VPS
   ├─ $20〜$100 → AWS / GCP / Fly.io
   └─ $100以上 → AWS / GCP / Azure（フルカスタム構成）
```

---

## 2. Vercel

### 2.1 概要とアーキテクチャ

Vercel は Next.js の開発元が提供するフロントエンド特化のクラウドプラットフォームである。Git push によるゼロコンフィグデプロイ、PR ごとのプレビュー環境自動作成、エッジネットワークによるグローバル配信を特徴とする。

```
Vercel アーキテクチャ:

  Git Push / PR
      │
      ▼
  ┌─────────────────┐
  │   Build System   │  ← フレームワーク自動検出
  │  (npm/pnpm/yarn) │     ビルドキャッシュ
  └────────┬─────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
  ┌──────┐   ┌──────────────┐
  │ CDN  │   │ Serverless   │
  │(静的) │   │  Functions   │
  │      │   │ (Node.js等)  │
  └──────┘   └──────────────┘
     │            │
     └─────┬──────┘
           ▼
  ┌─────────────────┐
  │  Edge Network   │  ← 世界中のエッジロケーション
  │  (グローバル配信) │
  └─────────────────┘
```

### 2.2 セットアップとデプロイフロー

```bash
# Vercel CLI のインストール
npm install -g vercel

# プロジェクトの初期化とリンク
vercel link

# 開発サーバー（Vercel の環境変数を利用可能）
vercel dev

# プレビューデプロイ（ステージング）
vercel

# 本番デプロイ
vercel --prod

# 環境変数の設定
vercel env add NEXT_PUBLIC_API_URL
vercel env add DATABASE_URL --sensitive

# 環境変数の一覧表示
vercel env ls

# 環境変数の削除
vercel env rm NEXT_PUBLIC_API_URL
```

### 2.3 vercel.json による高度な設定

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "framework": "nextjs",
  "buildCommand": "pnpm build",
  "installCommand": "pnpm install --frozen-lockfile",
  "outputDirectory": ".next",
  "regions": ["hnd1", "sfo1"],
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        { "key": "Access-Control-Allow-Origin", "value": "*" },
        { "key": "Access-Control-Allow-Methods", "value": "GET,POST,PUT,DELETE,OPTIONS" },
        { "key": "Access-Control-Allow-Headers", "value": "Content-Type, Authorization" },
        { "key": "Cache-Control", "value": "no-store, max-age=0" }
      ]
    },
    {
      "source": "/(.*)\\.(?:js|css|woff2?|png|jpg|svg|ico)",
      "headers": [
        { "key": "Cache-Control", "value": "public, max-age=31536000, immutable" }
      ]
    },
    {
      "source": "/(.*)",
      "headers": [
        { "key": "X-Frame-Options", "value": "DENY" },
        { "key": "X-Content-Type-Options", "value": "nosniff" },
        { "key": "Referrer-Policy", "value": "strict-origin-when-cross-origin" },
        {
          "key": "Content-Security-Policy",
          "value": "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:;"
        },
        { "key": "Permissions-Policy", "value": "camera=(), microphone=(), geolocation=()" }
      ]
    }
  ],
  "redirects": [
    { "source": "/old-page", "destination": "/new-page", "permanent": true },
    { "source": "/blog/:slug", "destination": "/posts/:slug", "permanent": true }
  ],
  "rewrites": [
    { "source": "/api/:path*", "destination": "https://api.example.com/:path*" }
  ],
  "functions": {
    "api/**/*.ts": {
      "memory": 1024,
      "maxDuration": 30
    },
    "api/heavy-task.ts": {
      "memory": 3009,
      "maxDuration": 60
    }
  },
  "crons": [
    {
      "path": "/api/cron/cleanup",
      "schedule": "0 0 * * *"
    },
    {
      "path": "/api/cron/sync",
      "schedule": "*/15 * * * *"
    }
  ]
}
```

### 2.4 Vercel Edge Functions

```typescript
// app/api/edge-example/route.ts
import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'edge';

// エッジで実行される API Route
export async function GET(request: NextRequest) {
  // リクエストのジオロケーション情報を取得
  const { geo, ip } = request;
  const country = geo?.country || 'Unknown';
  const city = geo?.city || 'Unknown';
  const region = geo?.region || 'Unknown';

  // ユーザーの地域に基づくコンテンツのパーソナライズ
  const greeting = getLocalizedGreeting(country);

  return NextResponse.json({
    greeting,
    location: { country, city, region },
    ip,
    timestamp: new Date().toISOString(),
    edge: true,
  }, {
    headers: {
      'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=300',
    },
  });
}

function getLocalizedGreeting(country: string): string {
  const greetings: Record<string, string> = {
    JP: 'こんにちは！',
    US: 'Hello!',
    KR: '안녕하세요!',
    CN: '你好！',
    FR: 'Bonjour!',
    DE: 'Hallo!',
    ES: '¡Hola!',
  };
  return greetings[country] || 'Hello!';
}
```

```typescript
// middleware.ts（Vercel Edge Middleware）
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const { pathname, searchParams } = request.nextUrl;
  const country = request.geo?.country || 'US';

  // 1. 地域ベースのリダイレクト
  if (pathname === '/' && country === 'JP') {
    return NextResponse.redirect(new URL('/ja', request.url));
  }

  // 2. A/Bテスト（Cookie ベース）
  const abTestCookie = request.cookies.get('ab-test-variant');
  if (!abTestCookie && pathname.startsWith('/landing')) {
    const variant = Math.random() < 0.5 ? 'a' : 'b';
    const response = NextResponse.rewrite(
      new URL(`/landing/${variant}`, request.url)
    );
    response.cookies.set('ab-test-variant', variant, {
      maxAge: 60 * 60 * 24 * 7, // 1週間
      httpOnly: true,
    });
    return response;
  }

  // 3. Bot検出とブロック
  const userAgent = request.headers.get('user-agent') || '';
  const isBot = /bot|crawl|spider|scrape/i.test(userAgent);
  if (isBot && pathname.startsWith('/api/')) {
    return new NextResponse('Forbidden', { status: 403 });
  }

  // 4. レート制限ヘッダーの付与
  const response = NextResponse.next();
  response.headers.set('X-Request-Country', country);
  response.headers.set('X-Request-Time', Date.now().toString());

  return response;
}

export const config = {
  matcher: ['/', '/landing/:path*', '/api/:path*'],
};
```

### 2.5 Vercel のプラン別機能と制限

```
Hobby プラン（無料）:
  ├─ 個人プロジェクト専用（商用利用不可）
  ├─ Serverless Functions: 10秒タイムアウト
  ├─ Edge Functions: 実行時間制限あり
  ├─ 帯域: 100GB/月
  ├─ ビルド時間: 6,000分/月
  ├─ 同時ビルド: 1
  ├─ チームメンバー: 1名
  ├─ Analytics: 基本（Web Vitals）
  ├─ プレビューデプロイ: 制限あり
  └─ サポート: コミュニティのみ

Pro プラン（$20/月/メンバー）:
  ├─ 商用利用可
  ├─ Serverless Functions: 60秒タイムアウト（最大300秒に拡張可能）
  ├─ Edge Functions: 拡張された実行時間
  ├─ 帯域: 1TB/月（超過分 $40/100GB）
  ├─ ビルド時間: 24,000分/月
  ├─ 同時ビルド: 3
  ├─ チームメンバー: 無制限
  ├─ Analytics: 高度（カスタムイベント対応）
  ├─ プレビューデプロイ: 無制限
  ├─ パスワード保護: ○
  ├─ DDoS 軽減: 基本
  ├─ サポート: メール
  └─ SAML SSO: 追加料金

Enterprise プラン（カスタム料金）:
  ├─ SLA 99.99%
  ├─ Serverless Functions: 900秒タイムアウト
  ├─ 帯域: カスタム
  ├─ 同時ビルド: カスタム
  ├─ 専用サポート
  ├─ SOC 2 / HIPAA 対応
  ├─ IP アクセス制限
  ├─ ファイアウォール
  ├─ マルチリージョン
  └─ カスタム SLA
```

### 2.6 Vercel のベストプラクティス

```typescript
// next.config.ts - Vercel 向け最適化設定
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // standalone 出力（Docker デプロイ時に有用）
  // output: 'standalone',

  // 画像最適化
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**.example.com',
      },
    ],
    formats: ['image/avif', 'image/webp'],
    minimumCacheTTL: 60 * 60 * 24 * 30, // 30日
  },

  // ヘッダー設定
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload',
          },
        ],
      },
    ];
  },

  // 実験的機能
  experimental: {
    // PPR (Partial Prerendering) の有効化
    ppr: true,
    // Server Actions の最適化
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },

  // ログ設定（Vercel でのデバッグ用）
  logging: {
    fetches: {
      fullUrl: true,
    },
  },
};

export default nextConfig;
```

### 2.7 Vercel のアンチパターン

```
アンチパターン 1: 大量の Serverless Functions
  問題: 関数ごとにコールドスタートが発生、コスト増
  対策: Route Handler を適切にグルーピングし、共通処理を Middleware に移動

アンチパターン 2: 巨大な node_modules のバンドル
  問題: ビルド時間の増大、関数サイズの肥大化
  対策: 不要な依存関係の削除、tree-shaking の活用、外部パッケージの最適化

アンチパターン 3: 環境変数の直書き
  問題: セキュリティリスク、環境間の設定混乱
  対策: Vercel のEnvironment Variables機能を使用、.env.local はgitignore

アンチパターン 4: ISR の revalidate 値が短すぎる
  問題: オリジンへのリクエスト集中、コスト増
  対策: コンテンツの更新頻度に応じた適切な revalidate 値の設定

アンチパターン 5: Preview デプロイの放置
  問題: 古いデプロイメントがリソースを消費
  対策: GitHub Actions で古いデプロイメントを自動クリーンアップ

アンチパターン 6: Hobby プランでの商用運用
  問題: 利用規約違反、突然のサービス停止リスク
  対策: 商用プロジェクトは必ず Pro プラン以上を使用
```

---

## 3. Cloudflare Pages / Workers

### 3.1 概要とアーキテクチャ

Cloudflare は世界 300 以上のエッジロケーションを持つ CDN/エッジコンピューティングプラットフォームである。V8 Isolates ベースの Workers によりコールドスタートなしのサーバーレス実行が可能で、Pages による静的サイトホスティングと組み合わせることでフルスタックのWebアプリケーションをエッジで配信できる。

```
Cloudflare アーキテクチャ:

  ユーザーリクエスト
      │
      ▼
  ┌─────────────────────┐
  │   Cloudflare Edge   │  ← 最寄りのエッジロケーション
  │   (300+ locations)   │
  └────────┬────────────┘
           │
     ┌─────┴──────────┐
     │                │
     ▼                ▼
  ┌──────────┐   ┌──────────────┐
  │  Pages   │   │   Workers    │
  │ (静的    │   │ (V8 Isolates)│
  │  アセット) │   │              │
  └──────────┘   └──────┬───────┘
                        │
              ┌─────────┼─────────┐
              │         │         │
              ▼         ▼         ▼
          ┌──────┐  ┌──────┐  ┌──────┐
          │  KV  │  │  D1  │  │  R2  │
          │      │  │(SQLite│  │(S3互  │
          │      │  │ 互換) │  │ 換)   │
          └──────┘  └──────┘  └──────┘
              │
              ▼
          ┌──────────────┐
          │ Durable      │
          │ Objects      │
          │ (ステート管理) │
          └──────────────┘
```

### 3.2 Cloudflare Pages のセットアップ

```bash
# Wrangler CLI のインストール
npm install -g wrangler

# 認証
wrangler login

# 新規プロジェクトの作成（フレームワーク選択）
npm create cloudflare@latest my-app

# 既存プロジェクトの Pages デプロイ
wrangler pages deploy ./dist

# Pages プロジェクトの作成（GitHub 連携）
wrangler pages project create my-project

# ローカル開発サーバー
wrangler pages dev ./dist --port 3000

# 環境変数の設定
wrangler pages secret put API_KEY
```

### 3.3 Workers の実装パターン

```typescript
// src/worker.ts - 基本的な Worker
export interface Env {
  MY_KV: KVNamespace;
  MY_DB: D1Database;
  MY_BUCKET: R2Bucket;
  API_KEY: string;
}

export default {
  async fetch(
    request: Request,
    env: Env,
    ctx: ExecutionContext
  ): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;

    // ルーティング
    switch (true) {
      case path === '/api/health':
        return new Response(JSON.stringify({ status: 'ok' }), {
          headers: { 'Content-Type': 'application/json' },
        });

      case path.startsWith('/api/users'):
        return handleUsers(request, env, ctx);

      case path.startsWith('/api/files'):
        return handleFiles(request, env, ctx);

      default:
        return new Response('Not Found', { status: 404 });
    }
  },

  // スケジュールされたタスク（Cron Triggers）
  async scheduled(
    event: ScheduledEvent,
    env: Env,
    ctx: ExecutionContext
  ): Promise<void> {
    ctx.waitUntil(cleanupExpiredData(env));
  },

  // キューの処理
  async queue(
    batch: MessageBatch<unknown>,
    env: Env,
    ctx: ExecutionContext
  ): Promise<void> {
    for (const message of batch.messages) {
      await processMessage(message, env);
      message.ack();
    }
  },
};

// ユーザー API のハンドラ
async function handleUsers(
  request: Request,
  env: Env,
  ctx: ExecutionContext
): Promise<Response> {
  const { method } = request;

  if (method === 'GET') {
    // D1 データベースからユーザー一覧を取得
    const { results } = await env.MY_DB.prepare(
      'SELECT id, name, email, created_at FROM users ORDER BY created_at DESC LIMIT 50'
    ).all();

    return Response.json({ users: results });
  }

  if (method === 'POST') {
    const body = await request.json() as { name: string; email: string };
    const { name, email } = body;

    // バリデーション
    if (!name || !email) {
      return Response.json(
        { error: 'name and email are required' },
        { status: 400 }
      );
    }

    // D1 にユーザーを挿入
    const result = await env.MY_DB.prepare(
      'INSERT INTO users (name, email, created_at) VALUES (?, ?, datetime("now")) RETURNING *'
    ).bind(name, email).first();

    // KV にキャッシュ
    ctx.waitUntil(
      env.MY_KV.put(`user:${result?.id}`, JSON.stringify(result), {
        expirationTtl: 3600,
      })
    );

    return Response.json({ user: result }, { status: 201 });
  }

  return new Response('Method Not Allowed', { status: 405 });
}

// ファイル処理ハンドラ（R2 使用）
async function handleFiles(
  request: Request,
  env: Env,
  ctx: ExecutionContext
): Promise<Response> {
  const url = new URL(request.url);
  const key = url.pathname.replace('/api/files/', '');

  if (request.method === 'PUT') {
    // R2 にファイルをアップロード
    const body = await request.arrayBuffer();
    const contentType = request.headers.get('content-type') || 'application/octet-stream';

    await env.MY_BUCKET.put(key, body, {
      httpMetadata: { contentType },
      customMetadata: {
        uploadedAt: new Date().toISOString(),
        size: body.byteLength.toString(),
      },
    });

    return Response.json({ key, size: body.byteLength });
  }

  if (request.method === 'GET') {
    const object = await env.MY_BUCKET.get(key);
    if (!object) {
      return new Response('Not Found', { status: 404 });
    }

    const headers = new Headers();
    headers.set('Content-Type', object.httpMetadata?.contentType || 'application/octet-stream');
    headers.set('Cache-Control', 'public, max-age=31536000, immutable');
    headers.set('ETag', object.httpEtag);

    return new Response(object.body, { headers });
  }

  return new Response('Method Not Allowed', { status: 405 });
}

async function cleanupExpiredData(env: Env): Promise<void> {
  await env.MY_DB.prepare(
    'DELETE FROM sessions WHERE expires_at < datetime("now")'
  ).run();
}

async function processMessage(
  message: Message<unknown>,
  env: Env
): Promise<void> {
  console.log('Processing message:', message.body);
}
```

### 3.4 wrangler.toml 設定

```toml
# wrangler.toml
name = "my-worker"
main = "src/worker.ts"
compatibility_date = "2024-12-01"
compatibility_flags = ["nodejs_compat"]

# 環境設定
[vars]
ENVIRONMENT = "production"

# KV Namespace
[[kv_namespaces]]
binding = "MY_KV"
id = "xxxxxxxxxxxxxxxxxxxx"
preview_id = "yyyyyyyyyyyyyyyyyyyy"

# D1 Database
[[d1_databases]]
binding = "MY_DB"
database_name = "my-database"
database_id = "zzzzzzzzzzzzzzzzzzzz"

# R2 Bucket
[[r2_buckets]]
binding = "MY_BUCKET"
bucket_name = "my-bucket"

# Cron Triggers
[triggers]
crons = ["0 0 * * *", "*/30 * * * *"]

# Queue
[[queues.producers]]
queue = "my-queue"
binding = "MY_QUEUE"

[[queues.consumers]]
queue = "my-queue"
max_batch_size = 10
max_batch_timeout = 30

# ルーティング
[[routes]]
pattern = "api.example.com/*"
zone_name = "example.com"

# ステージング環境
[env.staging]
name = "my-worker-staging"
[env.staging.vars]
ENVIRONMENT = "staging"

# D1 のマイグレーション
[[migrations]]
tag = "v1"
new_classes = ["UserDurableObject"]
```

### 3.5 Cloudflare D1 によるデータベース管理

```bash
# D1 データベースの作成
wrangler d1 create my-database

# マイグレーションの作成
wrangler d1 migrations create my-database init

# マイグレーション SQL の例
# migrations/0001_init.sql
```

```sql
-- migrations/0001_init.sql
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created_at ON users(created_at);

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS posts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  slug TEXT NOT NULL UNIQUE,
  published BOOLEAN NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_posts_user_id ON posts(user_id);
CREATE INDEX idx_posts_slug ON posts(slug);
CREATE INDEX idx_posts_published ON posts(published);
```

```bash
# マイグレーションの実行（ローカル）
wrangler d1 migrations apply my-database --local

# マイグレーションの実行（本番）
wrangler d1 migrations apply my-database --remote

# データベースへのクエリ実行
wrangler d1 execute my-database --command "SELECT * FROM users LIMIT 10"

# SQL ファイルの実行
wrangler d1 execute my-database --file ./seed.sql --remote
```

### 3.6 Cloudflare のプラン別機能と制限

```
Free プラン:
  Workers:
  ├─ リクエスト: 100,000/日
  ├─ CPU 時間: 10ms/リクエスト
  ├─ メモリ: 128MB
  └─ スクリプトサイズ: 1MB (圧縮後)

  Pages:
  ├─ ビルド: 500回/月
  ├─ 同時ビルド: 1
  ├─ 帯域: 無制限
  └─ サイト数: 無制限

  D1:
  ├─ 読み取り: 500万行/日
  ├─ 書き込み: 100,000行/日
  └─ ストレージ: 5GB

  KV:
  ├─ 読み取り: 100,000/日
  ├─ 書き込み: 1,000/日
  └─ ストレージ: 1GB

  R2:
  ├─ ストレージ: 10GB
  ├─ Class A 操作: 100万/月
  └─ Class B 操作: 1000万/月

Paid プラン ($5/月):
  Workers:
  ├─ リクエスト: 1000万/月（超過 $0.50/100万）
  ├─ CPU 時間: 50ms/リクエスト（Bundled）/ 30秒（Unbound）
  ├─ メモリ: 128MB
  └─ スクリプトサイズ: 10MB (圧縮後)

  D1:
  ├─ 読み取り: 250億行/月
  ├─ 書き込み: 5000万行/月
  └─ ストレージ: 50GB

  KV:
  ├─ 読み取り: 1000万/月
  ├─ 書き込み: 100万/月
  └─ ストレージ: 無制限
```

### 3.7 Cloudflare のベストプラクティスとアンチパターン

```
ベストプラクティス:
  1. KV はキャッシュとして使い、D1 を永続ストアにする
  2. waitUntil() で非同期処理をバックグラウンドに逃がす
  3. Workers の CPU 時間制限を意識した軽量な処理設計
  4. Durable Objects でステートフルな処理を実装
  5. Pages Functions で静的サイトに API を追加
  6. R2 でファイルストレージを S3 互換で構築

アンチパターン:
  1. Workers で重い計算処理を実行（CPU時間制限に抵触）
  2. KV に頻繁な書き込み（結果整合性のため不整合リスク）
  3. D1 で大量のJOINを含む複雑なクエリ（SQLite ベースのため制限あり）
  4. Node.js API への完全な依存（V8 Isolates は Node.js ランタイムではない）
  5. 単一の Worker に全ロジックを詰め込む（Service Bindings を活用）
```

---

## 4. Docker + セルフホスト

### 4.1 概要と利用シーン

Docker コンテナによるセルフホスティングは、プラットフォームロックインを回避し、インフラストラクチャの完全な制御を可能にする。VPS（Virtual Private Server）やオンプレミスサーバーに Docker をインストールし、アプリケーションをコンテナとして実行する。クラウドプロバイダへの依存を最小化できる反面、運用負荷が増大する点に注意が必要である。

```
Docker セルフホストの構成パターン:

パターン 1: シンプル（単一サーバー）
  VPS
  ├─ Docker
  │  ├─ App コンテナ（Next.js / Express）
  │  ├─ DB コンテナ（PostgreSQL）
  │  └─ Redis コンテナ（キャッシュ）
  ├─ Nginx（リバースプロキシ）
  └─ Let's Encrypt（SSL）

パターン 2: Docker Compose（複数サービス）
  VPS
  ├─ docker-compose.yml
  │  ├─ app（Next.js）
  │  ├─ api（Express/Fastify）
  │  ├─ db（PostgreSQL）
  │  ├─ redis（キャッシュ/セッション）
  │  ├─ nginx（リバースプロキシ）
  │  └─ certbot（SSL自動更新）
  └─ volumes/（永続データ）

パターン 3: Kubernetes（大規模）
  K8s クラスタ
  ├─ Deployments
  │  ├─ app（3 replicas）
  │  └─ api（3 replicas）
  ├─ Services
  ├─ Ingress（Nginx Ingress Controller）
  ├─ ConfigMaps / Secrets
  ├─ PersistentVolumeClaims
  └─ HPA（Horizontal Pod Autoscaler）
```

### 4.2 Next.js 本番用 Dockerfile（マルチステージビルド）

```dockerfile
# ============================================
# Next.js 本番用 Dockerfile（最適化版）
# ============================================

# ── Stage 1: 依存関係のインストール ──
FROM node:20-alpine AS deps
WORKDIR /app

# セキュリティ: 不要なパッケージを入れない
RUN apk add --no-cache libc6-compat

# パッケージマネージャのファイルをコピー
COPY package.json pnpm-lock.yaml ./

# pnpm を有効化して依存関係をインストール
RUN corepack enable pnpm && \
    pnpm install --frozen-lockfile --prod=false

# ── Stage 2: ビルド ──
FROM node:20-alpine AS builder
WORKDIR /app

# 環境変数（ビルド時に必要なもの）
ARG NEXT_PUBLIC_API_URL
ARG NEXT_PUBLIC_SITE_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_SITE_URL=$NEXT_PUBLIC_SITE_URL

# 依存関係をコピー
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# テレメトリの無効化
ENV NEXT_TELEMETRY_DISABLED=1

# ビルド実行
RUN corepack enable pnpm && pnpm build

# ── Stage 3: 本番イメージ ──
FROM node:20-alpine AS runner
WORKDIR /app

# 本番モード
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# セキュリティ: 非 root ユーザーで実行
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# 必要なファイルのみコピー（standalone 出力の場合）
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# ファイルの所有権を変更
RUN chown -R nextjs:nodejs /app

# 非 root ユーザーに切り替え
USER nextjs

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/api/health || exit 1

# ポートの公開
EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# サーバー起動
CMD ["node", "server.js"]
```

```javascript
// next.config.js（standalone 出力を有効化）
/** @type {import('next').NextConfig} */
module.exports = {
  output: 'standalone',
  // イメージサイズ削減のため不要な機能を無効化
  poweredByHeader: false,
  compress: true,
};
```

### 4.3 Express / Fastify API 用 Dockerfile

```dockerfile
# ============================================
# Express/Fastify API 本番用 Dockerfile
# ============================================

FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN corepack enable pnpm && pnpm install --frozen-lockfile

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN corepack enable pnpm && pnpm build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 appuser

# 本番用依存関係のみコピー
COPY --from=deps /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./

# セキュリティ: 不要なファイルを削除
RUN rm -rf /app/node_modules/.cache

USER appuser
EXPOSE 8080
ENV PORT=8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

CMD ["node", "dist/server.js"]
```

### 4.4 Docker Compose による本番構成

```yaml
# docker-compose.prod.yml
version: '3.9'

services:
  # ── アプリケーション ──
  app:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        NEXT_PUBLIC_API_URL: https://api.example.com
    restart: always
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://app:secret@db:5432/myapp
      - REDIS_URL=redis://redis:6379
      - SESSION_SECRET=${SESSION_SECRET}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - app-network
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 128M
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # ── データベース ──
  db:
    image: postgres:16-alpine
    restart: always
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: app
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d myapp"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - app-network
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G

  # ── Redis ──
  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - app-network
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 256M

  # ── Nginx リバースプロキシ ──
  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - certbot_data:/var/www/certbot:ro
      - certbot_certs:/etc/letsencrypt:ro
    depends_on:
      - app
    networks:
      - app-network

  # ── SSL 証明書の自動更新 ──
  certbot:
    image: certbot/certbot
    volumes:
      - certbot_data:/var/www/certbot
      - certbot_certs:/etc/letsencrypt
    entrypoint: "/bin/sh -c 'trap exit TERM; while :; do certbot renew; sleep 12h & wait $${!}; done;'"

volumes:
  postgres_data:
  redis_data:
  certbot_data:
  certbot_certs:

networks:
  app-network:
    driver: bridge
```

### 4.5 Nginx リバースプロキシ設定

```nginx
# nginx/conf.d/default.conf
upstream app_server {
    server app:3000;
    keepalive 64;
}

# HTTP → HTTPS リダイレクト
server {
    listen 80;
    server_name example.com www.example.com;

    # Let's Encrypt の認証用
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

# HTTPS サーバー
server {
    listen 443 ssl http2;
    server_name example.com www.example.com;

    # SSL 証明書
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    # SSL 設定
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;

    # セキュリティヘッダー
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Gzip 圧縮
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml application/json application/javascript
               application/xml+rss text/javascript image/svg+xml;

    # 静的ファイルのキャッシュ
    location /_next/static/ {
        proxy_pass http://app_server;
        proxy_cache_valid 200 365d;
        add_header Cache-Control "public, max-age=31536000, immutable";
    }

    # 画像のキャッシュ
    location ~* \.(jpg|jpeg|png|gif|ico|svg|webp|avif)$ {
        proxy_pass http://app_server;
        proxy_cache_valid 200 30d;
        add_header Cache-Control "public, max-age=2592000";
    }

    # API エンドポイント
    location /api/ {
        proxy_pass http://app_server;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;

        # レート制限
        limit_req zone=api burst=20 nodelay;
    }

    # メインアプリケーション
    location / {
        proxy_pass http://app_server;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### 4.6 Docker イメージの最適化テクニック

```
Docker イメージサイズ最適化:

手法 1: マルチステージビルド
  効果: ビルドツールを最終イメージに含めない
  目安: 1GB → 150MB への削減が可能

手法 2: Alpine ベースイメージ
  node:20        → 約 1GB
  node:20-slim   → 約 200MB
  node:20-alpine → 約 130MB

手法 3: .dockerignore の活用
  除外すべきもの:
  ├─ node_modules/
  ├─ .git/
  ├─ .next/
  ├─ .env*
  ├─ *.md
  ├─ tests/
  ├─ coverage/
  └─ .vscode/

手法 4: レイヤーキャッシュの最適化
  変更頻度の低いもの → 上位レイヤー
  変更頻度の高いもの → 下位レイヤー
  例: package.json のコピー → npm install → ソースコードのコピー

手法 5: standalone 出力（Next.js）
  通常: node_modules 全体が必要 → 数百MB
  standalone: 必要な依存のみバンドル → 50-100MB
```

```
# .dockerignore
node_modules
.git
.gitignore
.next
.env*
*.md
README*
LICENSE
tests/
__tests__/
coverage/
.vscode/
.idea/
.husky/
.github/
docker-compose*.yml
Dockerfile*
```

### 4.7 セルフホストのベストプラクティスとアンチパターン

```
ベストプラクティス:
  1. 非 root ユーザーでコンテナを実行する
  2. マルチステージビルドでイメージサイズを最小化
  3. HEALTHCHECK を必ず設定する
  4. ボリュームで永続データを管理する
  5. ログローテーションを設定する（max-size, max-file）
  6. リソース制限を設定する（CPU, メモリ）
  7. secrets はファイルマウントか環境変数で管理
  8. 定期的にベースイメージを更新する（セキュリティパッチ）

アンチパターン:
  1. root ユーザーでコンテナ実行 → 権限昇格のリスク
  2. latest タグの使用 → 再現性がない
  3. .env ファイルをイメージに含める → シークレットの漏洩
  4. ボリュームなしでDBを運用 → コンテナ削除でデータ消失
  5. ログをコンテナ内に保存 → ディスク圧迫
  6. リソース制限なし → OOM でホスト全体が影響
  7. ヘルスチェックなし → 障害検知の遅延
```

---

## 5. AWS デプロイパターン

### 5.1 AWS サービス選定マトリクス

```
AWS デプロイの4つのパターン:

                    Amplify      ECS Fargate    Lambda          S3+CloudFront
────────────────────────────────────────────────────────────────────────
対象              フルスタック    コンテナ       関数            静的サイト
SSR               ○            ○              △(Adapter必要)   ×
SSG               ○            ○              ○               ○
API               ○            ○              ○(API GW連携)    ×
コスト            低〜中        中〜高         最低〜低         最低
スケーリング      自動          自動(Fargate)   自動             自動
デプロイ速度      速い          やや遅い       速い             速い
カスタマイズ      低            高             中               低
学習コスト        低            中〜高         中               低
Docker必須        ×            ○              ×               ×
コールドスタート  なし          なし           あり             なし
最大実行時間      制限あり      無制限         15分             N/A
```

### 5.2 AWS Amplify によるデプロイ

```bash
# Amplify CLI のインストール
npm install -g @aws-amplify/cli

# Amplify の初期化
amplify init

# ホスティングの追加
amplify add hosting

# デプロイ
amplify publish
```

```yaml
# amplify.yml - Amplify ビルド設定
version: 1
frontend:
  phases:
    preBuild:
      commands:
        - corepack enable pnpm
        - pnpm install --frozen-lockfile
    build:
      commands:
        - pnpm build
  artifacts:
    baseDirectory: .next
    files:
      - '**/*'
  cache:
    paths:
      - node_modules/**/*
      - .next/cache/**/*

  # 環境変数（Amplify コンソールで設定推奨）
  environmentVariables:
    NEXT_PUBLIC_API_URL: https://api.example.com
```

### 5.3 AWS ECS Fargate によるコンテナデプロイ

```json
// task-definition.json（ECS タスク定義）
{
  "family": "my-nextjs-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "nextjs-app",
      "image": "ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:latest",
      "portMappings": [
        {
          "containerPort": 3000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        { "name": "NODE_ENV", "value": "production" },
        { "name": "PORT", "value": "3000" }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:ap-northeast-1:ACCOUNT_ID:secret:myapp/database-url"
        },
        {
          "name": "SESSION_SECRET",
          "valueFrom": "arn:aws:secretsmanager:ap-northeast-1:ACCOUNT_ID:secret:myapp/session-secret"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/my-nextjs-app",
          "awslogs-region": "ap-northeast-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:3000/api/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "essential": true
    }
  ]
}
```

```bash
# ECR リポジトリの作成
aws ecr create-repository --repository-name my-app --region ap-northeast-1

# Docker イメージのビルドとプッシュ
aws ecr get-login-password --region ap-northeast-1 | \
  docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com

docker build -t my-app .
docker tag my-app:latest ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:latest
docker push ACCOUNT_ID.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:latest

# ECS サービスの作成（Fargate）
aws ecs create-service \
  --cluster my-cluster \
  --service-name my-nextjs-service \
  --task-definition my-nextjs-app \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-zzz],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:ap-northeast-1:ACCOUNT_ID:targetgroup/my-tg/xxx,containerName=nextjs-app,containerPort=3000"

# サービスの更新（新しいイメージでデプロイ）
aws ecs update-service \
  --cluster my-cluster \
  --service my-nextjs-service \
  --force-new-deployment
```

### 5.4 AWS Lambda + API Gateway

```typescript
// lambda/handler.ts - Lambda 関数ハンドラ
import { APIGatewayProxyHandlerV2 } from 'aws-lambda';

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
  const { httpMethod, path, body, queryStringParameters, headers } = event;

  try {
    // ルーティング
    if (path === '/api/health') {
      return {
        statusCode: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'ok', timestamp: new Date().toISOString() }),
      };
    }

    if (path === '/api/users' && httpMethod === 'GET') {
      // DynamoDB からデータ取得（例）
      const users = await getUsers();
      return {
        statusCode: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'max-age=60',
        },
        body: JSON.stringify({ users }),
      };
    }

    if (path === '/api/users' && httpMethod === 'POST') {
      const userData = JSON.parse(body || '{}');
      const user = await createUser(userData);
      return {
        statusCode: 201,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user }),
      };
    }

    return {
      statusCode: 404,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Not Found' }),
    };
  } catch (error) {
    console.error('Lambda error:', error);
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Internal Server Error' }),
    };
  }
};

async function getUsers() {
  // DynamoDB / RDS からの取得ロジック
  return [];
}

async function createUser(data: Record<string, unknown>) {
  // ユーザー作成ロジック
  return { id: 'new-id', ...data };
}
```

### 5.5 S3 + CloudFront による静的サイト配信

```bash
# S3 バケットの作成（静的ウェブサイトホスティング用）
aws s3 mb s3://my-static-site --region ap-northeast-1

# バケットポリシーの設定（CloudFront OAI 用）
aws s3api put-bucket-policy --bucket my-static-site --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCloudFrontOAI",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::cloudfront:user/CloudFront Origin Access Identity XXXXXX"
      },
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::my-static-site/*"
    }
  ]
}'

# ビルド成果物のアップロード
aws s3 sync ./dist s3://my-static-site \
  --delete \
  --cache-control "public, max-age=31536000, immutable" \
  --exclude "*.html"

# HTML ファイルは短いキャッシュで
aws s3 sync ./dist s3://my-static-site \
  --include "*.html" \
  --cache-control "public, max-age=0, must-revalidate"

# CloudFront のキャッシュ無効化
aws cloudfront create-invalidation \
  --distribution-id EXXXXXX \
  --paths "/*"
```

### 5.6 AWS デプロイの CI/CD（GitHub Actions）

```yaml
# .github/workflows/deploy-aws.yml
name: Deploy to AWS

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: ap-northeast-1
  ECR_REPOSITORY: my-app
  ECS_SERVICE: my-nextjs-service
  ECS_CLUSTER: my-cluster
  CONTAINER_NAME: nextjs-app

permissions:
  id-token: write
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with:
          version: 9
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'pnpm'
      - run: pnpm install --frozen-lockfile
      - run: pnpm lint
      - run: pnpm test

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # AWS 認証（OIDC）
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::${{ secrets.AWS_ACCOUNT_ID }}:role/github-actions-role
          aws-region: ${{ env.AWS_REGION }}

      # ECR ログイン
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      # Docker ビルドとプッシュ
      - name: Build, tag, and push image to Amazon ECR
        id: build-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build \
            --build-arg NEXT_PUBLIC_API_URL=${{ secrets.NEXT_PUBLIC_API_URL }} \
            -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
            -t $ECR_REGISTRY/$ECR_REPOSITORY:latest \
            .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

      # ECS タスク定義の更新
      - name: Fill in the new image ID in the Amazon ECS task definition
        id: task-def
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: task-definition.json
          container-name: ${{ env.CONTAINER_NAME }}
          image: ${{ steps.build-image.outputs.image }}

      # ECS サービスのデプロイ
      - name: Deploy Amazon ECS task definition
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition: ${{ steps.task-def.outputs.task-definition }}
          service: ${{ env.ECS_SERVICE }}
          cluster: ${{ env.ECS_CLUSTER }}
          wait-for-service-stability: true
```

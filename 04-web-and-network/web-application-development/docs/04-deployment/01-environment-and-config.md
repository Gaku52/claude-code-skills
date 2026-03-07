# 環境設定

> 環境設定はアプリケーションの安全な運用の基盤。環境変数の管理、Feature Flags、設定の階層化、シークレット管理まで、本番環境で安全に設定を管理するベストプラクティスを習得する。

## この章で学ぶこと

- [ ] 環境変数の設計と安全な管理を理解する
- [ ] 環境ごとの設定分離パターンを習得する
- [ ] Feature Flags の実装と運用を把握する
- [ ] シークレット管理のベストプラクティスを学ぶ
- [ ] 12-Factor App に基づく設定管理を理解する
- [ ] CI/CD パイプラインにおける環境変数の受け渡しを習得する
- [ ] 設定のバリデーションとフェイルファスト戦略を実装する
- [ ] マルチ環境運用でのトラブルシューティング手法を身につける

---

## 1. 環境設定の基本概念

### 1.1 なぜ環境設定が重要なのか

アプリケーション開発において、環境設定（Configuration Management）は見過ごされがちだが、セキュリティ、運用安定性、開発効率のすべてに直結する極めて重要な領域である。

```
環境設定が引き起こすインシデントの例:

  ケース1: 本番 DB のクレデンシャルが GitHub に流出
    原因: .env ファイルを誤って Git にコミット
    影響: データベース全体が外部に露出、ユーザーデータ漏洩
    対策: .gitignore の徹底、pre-commit hook でのチェック

  ケース2: ステージング環境が本番 API を呼び出す
    原因: 環境変数の設定ミスで API_URL が本番を指していた
    影響: テストデータが本番に混入、顧客に影響
    対策: 環境別設定の厳密な分離、起動時バリデーション

  ケース3: Feature Flag の設定漏れで未完成機能が本番に露出
    原因: デプロイ時に Feature Flag の環境変数を設定し忘れた
    影響: 未完成の UI がユーザーに表示、バグ報告殺到
    対策: デフォルト値の適切な設定、デプロイチェックリスト

  ケース4: 本番環境でデバッグモードが有効のまま
    原因: NODE_ENV が development のままデプロイされた
    影響: スタックトレースが外部に露出、セキュリティリスク
    対策: 起動時の環境チェック、CI/CD での自動検証
```

### 1.2 12-Factor App と設定管理

Heroku のエンジニアが提唱した 12-Factor App の原則において、設定（Config）は第3の要素として定義されている。この原則はモダンなクラウドネイティブアプリケーションの基礎となっている。

```
12-Factor App における設定の原則:

  ① 設定はコードから厳密に分離する
     - 環境変数として外部から注入
     - 設定ファイルをコードリポジトリに含めない
     - 同じコードベースがすべての環境で動作する

  ② 環境間の差異は設定のみで表現する
     - development / staging / production の違いは設定だけ
     - コードの条件分岐で環境を判定しない
     - ビルド成果物は全環境で同一

  ③ 設定はデプロイごとに変わりうる
     - コードの変更なしに設定を変更できる
     - 再デプロイなしでの設定変更が可能
     - 設定の変更履歴を追跡できる

12-Factor App の 12 の要素:
  I.    コードベース（一つのコードベース、複数のデプロイ）
  II.   依存関係（明示的に宣言し分離する）
  III.  設定（環境変数に格納する）← ★本章のテーマ
  IV.   バックエンドサービス（アタッチされたリソースとして扱う）
  V.    ビルド・リリース・実行（3つのステージを厳密に分離する）
  VI.   プロセス（ステートレスなプロセスとして実行する）
  VII.  ポートバインディング（ポートバインディングでサービスを公開する）
  VIII. 並行性（プロセスモデルでスケールアウトする）
  IX.   廃棄容易性（高速な起動とグレースフルシャットダウン）
  X.    開発/本番一致（開発・ステージング・本番を可能な限り一致させる）
  XI.   ログ（ログをイベントストリームとして扱う）
  XII.  管理プロセス（管理タスクをワンオフプロセスとして実行する）
```

### 1.3 環境の階層と種類

```
典型的な環境の階層:

  ┌──────────────────────────────────────────────────┐
  │  local (開発者のマシン)                           │
  │  ├── 個人の .env.local で設定を上書き             │
  │  ├── ホットリロード有効                           │
  │  └── デバッグツール有効                           │
  ├──────────────────────────────────────────────────┤
  │  development (共有開発環境)                       │
  │  ├── 開発チーム全体で共有するサーバー              │
  │  ├── テスト用の外部サービスに接続                  │
  │  └── 本番に近い構成だがリソースは最小限            │
  ├──────────────────────────────────────────────────┤
  │  staging (ステージング環境)                       │
  │  ├── 本番とほぼ同一の構成                         │
  │  ├── 本番データのサブセットで動作                  │
  │  ├── QA テスト、受け入れテストを実施               │
  │  └── 本番デプロイ前の最終確認                     │
  ├──────────────────────────────────────────────────┤
  │  production (本番環境)                            │
  │  ├── 実際のユーザーがアクセスする環境              │
  │  ├── 最高レベルのセキュリティ設定                  │
  │  ├── 監視・アラート完備                           │
  │  └── パフォーマンス最適化済み                     │
  └──────────────────────────────────────────────────┘

  追加の環境（大規模プロジェクトの場合）:
  ├── preview / PR環境: PR ごとに自動生成される一時的な環境
  ├── canary: 本番の一部トラフィックのみ受ける環境
  ├── sandbox: 外部パートナー向けのテスト環境
  └── disaster-recovery: 災害時に切り替える予備環境

.env ファイルの優先順位（Next.js の場合）:
  .env                  ← デフォルト（全環境共通）
  .env.local            ← ローカルのオーバーライド（.gitignore）
  .env.development      ← 開発環境固有
  .env.development.local← 開発環境のローカルオーバーライド
  .env.staging          ← ステージング固有
  .env.production       ← 本番環境固有
  .env.production.local ← 本番のローカルオーバーライド（.gitignore）

  読み込み優先順位（後勝ち）:
  .env < .env.local < .env.[NODE_ENV] < .env.[NODE_ENV].local
```

---

## 2. 環境変数の設計

### 2.1 命名規則とプレフィックス

環境変数の命名は一貫性を持たせることが重要である。明確な命名規則を定めることで、変数の用途が一目でわかり、設定ミスを防止できる。

```
命名規則の基本原則:

  ✅ 推奨パターン:
    SCREAMING_SNAKE_CASE を使用
    ├── DATABASE_URL          → データベース接続先
    ├── REDIS_HOST            → Redis ホスト名
    ├── AWS_ACCESS_KEY_ID     → AWS アクセスキー
    ├── SMTP_PORT             → メールサーバーポート
    └── LOG_LEVEL             → ログレベル

  ✅ プレフィックスによる分類:
    サービス別:
    ├── DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    ├── REDIS_URL, REDIS_PASSWORD, REDIS_DB
    ├── AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    ├── STRIPE_PUBLIC_KEY, STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET
    ├── SENDGRID_API_KEY, SENDGRID_FROM_EMAIL
    └── SENTRY_DSN, SENTRY_ENVIRONMENT

    機能別:
    ├── ENABLE_ANALYTICS      → 分析機能の有効/無効
    ├── ENABLE_RATE_LIMIT     → レート制限の有効/無効
    ├── IS_MAINTENANCE_MODE   → メンテナンスモード判定
    ├── HAS_PREMIUM_FEATURES  → プレミアム機能の有無
    └── MAX_UPLOAD_SIZE       → アップロード上限

    公開範囲別（Next.js）:
    ├── NEXT_PUBLIC_API_URL   → クライアントに公開
    ├── NEXT_PUBLIC_GA_ID     → Google Analytics ID
    ├── DATABASE_URL          → サーバーのみ（機密）
    └── JWT_SECRET            → サーバーのみ（機密）

  ❌ アンチパターン:
    ├── KEY           → 何のキーかわからない
    ├── SECRET        → 何のシークレットかわからない
    ├── URL           → 何の URL かわからない
    ├── password      → snake_case ではない
    ├── ApiKey        → camelCase は使わない
    └── my-config     → ハイフンは使えない（シェルで問題）

フレームワーク別の公開プレフィックス:
  ┌─────────────────┬──────────────────────┬───────────────────┐
  │ フレームワーク    │ 公開プレフィックス     │ 非公開（サーバー）  │
  ├─────────────────┼──────────────────────┼───────────────────┤
  │ Next.js         │ NEXT_PUBLIC_         │ プレフィックスなし  │
  │ Vite            │ VITE_                │ プレフィックスなし  │
  │ Create React App│ REACT_APP_           │ プレフィックスなし  │
  │ Nuxt.js         │ NUXT_PUBLIC_         │ NUXT_ or なし     │
  │ SvelteKit       │ PUBLIC_              │ プレフィックスなし  │
  │ Remix           │ なし（全てサーバー）   │ 全て              │
  │ Astro           │ PUBLIC_              │ プレフィックスなし  │
  └─────────────────┴──────────────────────┴───────────────────┘
```

### 2.2 型安全な環境変数管理（Zod）

TypeScript プロジェクトでは、Zod を使って環境変数にスキーマバリデーションを適用することで、型安全性を確保し、起動時に設定ミスを早期検出できる。

```typescript
// ============================================
// config/env.ts - 型安全な環境変数管理
// ============================================
import { z } from 'zod';

// ---- サーバーサイド環境変数スキーマ ----
const serverEnvSchema = z.object({
  // アプリケーション基本設定
  NODE_ENV: z.enum(['development', 'staging', 'production', 'test'])
    .default('development'),
  PORT: z.coerce.number().int().min(1).max(65535).default(3000),
  HOST: z.string().default('0.0.0.0'),

  // データベース
  DATABASE_URL: z.string().url()
    .refine(url => url.startsWith('postgresql://') || url.startsWith('postgres://'), {
      message: 'DATABASE_URL must be a PostgreSQL connection string',
    }),
  DATABASE_POOL_MIN: z.coerce.number().int().min(1).default(2),
  DATABASE_POOL_MAX: z.coerce.number().int().min(1).default(10),
  DATABASE_SSL: z.coerce.boolean().default(true),

  // Redis
  REDIS_URL: z.string().url().optional(),
  REDIS_PASSWORD: z.string().optional(),
  REDIS_DB: z.coerce.number().int().min(0).max(15).default(0),

  // 認証
  JWT_SECRET: z.string().min(32, 'JWT_SECRET must be at least 32 characters'),
  JWT_EXPIRES_IN: z.string().default('7d'),
  SESSION_SECRET: z.string().min(32).optional(),

  // 外部サービス
  STRIPE_SECRET_KEY: z.string().startsWith('sk_'),
  STRIPE_WEBHOOK_SECRET: z.string().startsWith('whsec_'),
  SENDGRID_API_KEY: z.string().startsWith('SG.'),
  SENTRY_DSN: z.string().url().optional(),

  // AWS
  AWS_REGION: z.string().default('ap-northeast-1'),
  AWS_ACCESS_KEY_ID: z.string().optional(),
  AWS_SECRET_ACCESS_KEY: z.string().optional(),
  S3_BUCKET_NAME: z.string().optional(),

  // 機能フラグ
  ENABLE_RATE_LIMIT: z.coerce.boolean().default(true),
  ENABLE_CORS: z.coerce.boolean().default(true),
  ENABLE_SWAGGER: z.coerce.boolean().default(false),

  // ログ
  LOG_LEVEL: z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace'])
    .default('info'),
  LOG_FORMAT: z.enum(['json', 'pretty']).default('json'),
});

// ---- クライアントサイド環境変数スキーマ ----
const clientEnvSchema = z.object({
  NEXT_PUBLIC_API_URL: z.string().url(),
  NEXT_PUBLIC_APP_URL: z.string().url(),
  NEXT_PUBLIC_GA_ID: z.string().startsWith('G-').optional(),
  NEXT_PUBLIC_SENTRY_DSN: z.string().url().optional(),
  NEXT_PUBLIC_STRIPE_PUBLIC_KEY: z.string().startsWith('pk_'),
  NEXT_PUBLIC_ENABLE_ANALYTICS: z.coerce.boolean().default(false),
  NEXT_PUBLIC_APP_VERSION: z.string().default('0.0.0'),
});

// ---- バリデーションとエクスポート ----
function validateEnv<T extends z.ZodType>(
  schema: T,
  env: Record<string, string | undefined>,
  label: string
): z.infer<T> {
  const result = schema.safeParse(env);

  if (!result.success) {
    const errors = result.error.flatten().fieldErrors;
    const errorMessages = Object.entries(errors)
      .map(([key, msgs]) => `  ${key}: ${(msgs as string[]).join(', ')}`)
      .join('\n');

    console.error(`\n${label} の環境変数バリデーションエラー:\n${errorMessages}\n`);

    // 開発環境では詳細を表示、本番では最小限の情報で停止
    if (process.env.NODE_ENV === 'production') {
      console.error('本番環境の設定エラーにより起動を中止します。');
    } else {
      console.error('必要な環境変数を .env.local に設定してください。');
      console.error('参考: .env.example を確認してください。');
    }

    process.exit(1);
  }

  return result.data;
}

// サーバーサイドでのみ実行（クライアントバンドルに含めない）
export const serverEnv = typeof window === 'undefined'
  ? validateEnv(serverEnvSchema, process.env, 'サーバー')
  : ({} as z.infer<typeof serverEnvSchema>);

// クライアントサイドでも利用可能
export const clientEnv = validateEnv(clientEnvSchema, process.env, 'クライアント');

// 統合型のエクスポート
export const env = { ...serverEnv, ...clientEnv };

// 型のエクスポート
export type ServerEnv = z.infer<typeof serverEnvSchema>;
export type ClientEnv = z.infer<typeof clientEnvSchema>;
```

### 2.3 環境変数バリデーションのテスト

```typescript
// ============================================
// config/__tests__/env.test.ts
// ============================================
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { z } from 'zod';

// テスト用のスキーマ（本番と同じ定義を使用）
const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']),
  DATABASE_URL: z.string().url(),
  JWT_SECRET: z.string().min(32),
  PORT: z.coerce.number().int().min(1).max(65535).default(3000),
});

describe('環境変数バリデーション', () => {
  const validEnv = {
    NODE_ENV: 'production',
    DATABASE_URL: 'postgresql://user:pass@localhost:5432/mydb',
    JWT_SECRET: 'a-very-long-secret-key-that-is-at-least-32-characters',
    PORT: '8080',
  };

  it('有効な環境変数が正しくパースされること', () => {
    const result = envSchema.safeParse(validEnv);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.PORT).toBe(8080); // 文字列から数値に変換
      expect(result.data.NODE_ENV).toBe('production');
    }
  });

  it('無効な NODE_ENV が拒否されること', () => {
    const result = envSchema.safeParse({
      ...validEnv,
      NODE_ENV: 'invalid',
    });
    expect(result.success).toBe(false);
  });

  it('DATABASE_URL が URL 形式でない場合に拒否されること', () => {
    const result = envSchema.safeParse({
      ...validEnv,
      DATABASE_URL: 'not-a-url',
    });
    expect(result.success).toBe(false);
  });

  it('JWT_SECRET が短すぎる場合に拒否されること', () => {
    const result = envSchema.safeParse({
      ...validEnv,
      JWT_SECRET: 'short',
    });
    expect(result.success).toBe(false);
  });

  it('PORT のデフォルト値が適用されること', () => {
    const { PORT, ...envWithoutPort } = validEnv;
    const result = envSchema.safeParse(envWithoutPort);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.PORT).toBe(3000);
    }
  });

  it('PORT が範囲外の場合に拒否されること', () => {
    const result = envSchema.safeParse({
      ...validEnv,
      PORT: '70000',
    });
    expect(result.success).toBe(false);
  });
});
```

### 2.4 .env.example の管理

プロジェクトには必ず `.env.example` を含め、必要な環境変数の一覧とその説明を記載する。

```bash
# ============================================
# .env.example
# このファイルを .env.local にコピーして値を設定してください
# cp .env.example .env.local
# ============================================

# ---- アプリケーション基本設定 ----
NODE_ENV=development
PORT=3000
HOST=0.0.0.0

# ---- データベース ----
# PostgreSQL の接続文字列
# ローカル: postgresql://postgres:password@localhost:5432/myapp_dev
# Docker: postgresql://postgres:password@db:5432/myapp_dev
DATABASE_URL=postgresql://postgres:password@localhost:5432/myapp_dev
DATABASE_POOL_MIN=2
DATABASE_POOL_MAX=10
DATABASE_SSL=false

# ---- Redis ----
# ローカル: redis://localhost:6379
# Docker: redis://redis:6379
REDIS_URL=redis://localhost:6379
REDIS_DB=0

# ---- 認証 ----
# 32文字以上のランダムな文字列を設定
# 生成コマンド: openssl rand -base64 48
JWT_SECRET=your-jwt-secret-at-least-32-characters-long
JWT_EXPIRES_IN=7d

# ---- Stripe ----
# https://dashboard.stripe.com/test/apikeys からテストキーを取得
STRIPE_SECRET_KEY=sk_test_xxxxxxxxxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxx
NEXT_PUBLIC_STRIPE_PUBLIC_KEY=pk_test_xxxxxxxxxxxx

# ---- メール（SendGrid）----
# https://app.sendgrid.com/settings/api_keys から取得
SENDGRID_API_KEY=SG.xxxxxxxxxxxx

# ---- エラー監視（Sentry）----
# https://sentry.io から DSN を取得
SENTRY_DSN=https://xxxxx@sentry.io/xxxxx
NEXT_PUBLIC_SENTRY_DSN=https://xxxxx@sentry.io/xxxxx

# ---- AWS ----
AWS_REGION=ap-northeast-1
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
S3_BUCKET_NAME=myapp-dev-uploads

# ---- フロントエンド公開設定 ----
NEXT_PUBLIC_API_URL=http://localhost:3001
NEXT_PUBLIC_APP_URL=http://localhost:3000
NEXT_PUBLIC_GA_ID=G-XXXXXXXXXX
NEXT_PUBLIC_ENABLE_ANALYTICS=false
NEXT_PUBLIC_APP_VERSION=0.0.0

# ---- 機能フラグ ----
ENABLE_RATE_LIMIT=false
ENABLE_CORS=true
ENABLE_SWAGGER=true

# ---- ログ ----
LOG_LEVEL=debug
LOG_FORMAT=pretty
```

### 2.5 環境変数の自動チェックスクリプト

```typescript
// ============================================
// scripts/check-env.ts
// プロジェクト起動前に環境変数の整合性をチェックする
// ============================================
import fs from 'fs';
import path from 'path';

interface EnvCheckResult {
  missing: string[];
  empty: string[];
  warnings: string[];
}

function parseEnvFile(filePath: string): Map<string, string> {
  const envMap = new Map<string, string>();

  if (!fs.existsSync(filePath)) {
    return envMap;
  }

  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');

  for (const line of lines) {
    const trimmed = line.trim();
    // コメント行と空行をスキップ
    if (trimmed === '' || trimmed.startsWith('#')) continue;

    const equalIndex = trimmed.indexOf('=');
    if (equalIndex === -1) continue;

    const key = trimmed.substring(0, equalIndex).trim();
    const value = trimmed.substring(equalIndex + 1).trim();
    envMap.set(key, value);
  }

  return envMap;
}

function checkEnv(): EnvCheckResult {
  const projectRoot = process.cwd();
  const exampleEnv = parseEnvFile(path.join(projectRoot, '.env.example'));
  const localEnv = parseEnvFile(path.join(projectRoot, '.env.local'));
  const envFile = parseEnvFile(path.join(projectRoot, '.env'));

  // 実効的な環境変数: .env → .env.local → process.env
  const effectiveEnv = new Map([...envFile, ...localEnv]);

  const result: EnvCheckResult = {
    missing: [],
    empty: [],
    warnings: [],
  };

  for (const [key, exampleValue] of exampleEnv) {
    const actualValue = effectiveEnv.get(key) || process.env[key];

    if (actualValue === undefined) {
      result.missing.push(key);
    } else if (actualValue === '' || actualValue === exampleValue) {
      // 値が空またはサンプル値のまま
      if (key.includes('SECRET') || key.includes('KEY') || key.includes('PASSWORD')) {
        result.warnings.push(`${key}: サンプル値のまま or 空です（機密情報）`);
      } else if (actualValue === '') {
        result.empty.push(key);
      }
    }
  }

  // NEXT_PUBLIC_ でないのにクライアントで使おうとしている変数のチェック
  for (const [key] of effectiveEnv) {
    if (key.includes('SECRET') || key.includes('PASSWORD') || key.includes('PRIVATE')) {
      if (key.startsWith('NEXT_PUBLIC_')) {
        result.warnings.push(
          `${key}: 機密情報が NEXT_PUBLIC_ プレフィックスで公開されています！`
        );
      }
    }
  }

  return result;
}

// 実行
const result = checkEnv();

if (result.missing.length > 0) {
  console.error('\n未設定の環境変数:');
  result.missing.forEach(key => console.error(`  - ${key}`));
}

if (result.empty.length > 0) {
  console.warn('\n空の環境変数:');
  result.empty.forEach(key => console.warn(`  - ${key}`));
}

if (result.warnings.length > 0) {
  console.warn('\n警告:');
  result.warnings.forEach(msg => console.warn(`  - ${msg}`));
}

if (result.missing.length === 0 && result.warnings.length === 0) {
  console.log('\nすべての環境変数が正しく設定されています。');
}

// 必須の環境変数が不足している場合は終了コード 1 で終了
if (result.missing.length > 0) {
  process.exit(1);
}
```

### 2.6 各フレームワークでの環境変数の使い方

```typescript
// ============================================
// Next.js での環境変数の使い方
// ============================================

// 1. サーバーコンポーネント（App Router）
// サーバーサイドの環境変数に直接アクセス可能
async function ServerComponent() {
  // サーバーサイドのみ: OK
  const dbUrl = process.env.DATABASE_URL;
  const apiKey = process.env.STRIPE_SECRET_KEY;

  // NEXT_PUBLIC_ もサーバーで利用可能
  const publicApiUrl = process.env.NEXT_PUBLIC_API_URL;

  const data = await fetch(publicApiUrl + '/api/data', {
    headers: { Authorization: `Bearer ${apiKey}` },
  });

  return <div>{/* ... */}</div>;
}

// 2. クライアントコンポーネント
'use client';
function ClientComponent() {
  // クライアントサイドでは NEXT_PUBLIC_ のみアクセス可能
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;     // OK
  // const secret = process.env.DATABASE_URL;          // undefined（安全）

  return <div>API: {apiUrl}</div>;
}

// 3. API Route（App Router）
// app/api/webhook/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  // サーバーサイドなので全環境変数にアクセス可能
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

  // Webhook の署名検証
  const signature = request.headers.get('stripe-signature');
  // ...
  return NextResponse.json({ received: true });
}

// 4. next.config.js での環境変数
/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    // ビルド時に埋め込まれる（非推奨: NEXT_PUBLIC_ を使うべき）
    CUSTOM_VAR: process.env.CUSTOM_VAR,
  },
  // ランタイム設定（サーバーサイドのみ）
  serverRuntimeConfig: {
    apiSecret: process.env.API_SECRET,
  },
  // パブリックランタイム設定（クライアントからもアクセス可能）
  publicRuntimeConfig: {
    apiUrl: process.env.NEXT_PUBLIC_API_URL,
  },
};

module.exports = nextConfig;
```

```typescript
// ============================================
// Vite での環境変数の使い方
// ============================================

// vite.config.ts
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  // mode に応じた .env ファイルを読み込む
  const env = loadEnv(mode, process.cwd(), '');

  return {
    define: {
      // カスタム変数をグローバルに定義
      __APP_VERSION__: JSON.stringify(env.npm_package_version),
    },
    server: {
      port: parseInt(env.VITE_DEV_PORT || '5173'),
    },
  };
});

// コンポーネントでの使用
function App() {
  // VITE_ プレフィックスの変数のみアクセス可能
  const apiUrl = import.meta.env.VITE_API_URL;
  const mode = import.meta.env.MODE; // 'development' | 'production'
  const isDev = import.meta.env.DEV; // boolean
  const isProd = import.meta.env.PROD; // boolean

  return <div>API: {apiUrl}</div>;
}

// 型定義（env.d.ts）
/// <reference types="vite/client" />
interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_APP_TITLE: string;
  readonly VITE_GA_ID?: string;
  readonly VITE_ENABLE_MOCK: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

---

## 3. 設定の分離パターン

### 3.1 環境別設定ファイルパターン

```typescript
// ============================================
// config/index.ts - 環境別設定の統合管理
// ============================================

// 基本設定の型定義
interface AppConfig {
  app: {
    name: string;
    version: string;
    url: string;
  };
  api: {
    baseUrl: string;
    timeout: number;
    retryCount: number;
    retryDelay: number;
  };
  database: {
    poolMin: number;
    poolMax: number;
    ssl: boolean;
    logging: boolean;
  };
  cache: {
    ttl: number;
    maxItems: number;
    strategy: 'memory' | 'redis';
  };
  features: {
    analytics: boolean;
    debugMode: boolean;
    maintenanceMode: boolean;
    rateLimit: boolean;
  };
  security: {
    corsOrigins: string[];
    rateLimitWindow: number;
    rateLimitMax: number;
    csrfEnabled: boolean;
  };
  logging: {
    level: string;
    format: 'json' | 'pretty';
    destination: 'stdout' | 'file' | 'both';
  };
}

// 全環境共通のデフォルト設定
const defaultConfig: AppConfig = {
  app: {
    name: 'MyApp',
    version: process.env.npm_package_version || '0.0.0',
    url: 'http://localhost:3000',
  },
  api: {
    baseUrl: 'http://localhost:3001',
    timeout: 30000,
    retryCount: 3,
    retryDelay: 1000,
  },
  database: {
    poolMin: 2,
    poolMax: 10,
    ssl: false,
    logging: false,
  },
  cache: {
    ttl: 300,
    maxItems: 1000,
    strategy: 'memory',
  },
  features: {
    analytics: false,
    debugMode: false,
    maintenanceMode: false,
    rateLimit: true,
  },
  security: {
    corsOrigins: ['http://localhost:3000'],
    rateLimitWindow: 15 * 60 * 1000, // 15分
    rateLimitMax: 100,
    csrfEnabled: true,
  },
  logging: {
    level: 'info',
    format: 'json',
    destination: 'stdout',
  },
};

// 環境別のオーバーライド設定
const envOverrides: Record<string, Partial<AppConfig>> = {
  development: {
    api: {
      ...defaultConfig.api,
      timeout: 60000, // 開発時は長めのタイムアウト
    },
    database: {
      ...defaultConfig.database,
      logging: true, // SQL ログ出力
    },
    cache: {
      ...defaultConfig.cache,
      ttl: 0, // 開発時はキャッシュ無効
    },
    features: {
      ...defaultConfig.features,
      debugMode: true,
      rateLimit: false, // 開発時はレート制限なし
    },
    logging: {
      level: 'debug',
      format: 'pretty',
      destination: 'stdout',
    },
  },
  staging: {
    app: {
      ...defaultConfig.app,
      url: 'https://staging.example.com',
    },
    api: {
      ...defaultConfig.api,
      baseUrl: 'https://staging-api.example.com',
      timeout: 15000,
    },
    database: {
      ...defaultConfig.database,
      ssl: true,
    },
    cache: {
      ...defaultConfig.cache,
      ttl: 60,
      strategy: 'redis',
    },
    features: {
      ...defaultConfig.features,
      analytics: true,
      debugMode: true,
    },
    security: {
      ...defaultConfig.security,
      corsOrigins: ['https://staging.example.com'],
    },
  },
  production: {
    app: {
      ...defaultConfig.app,
      url: 'https://www.example.com',
    },
    api: {
      ...defaultConfig.api,
      baseUrl: 'https://api.example.com',
      timeout: 10000,
      retryCount: 5,
    },
    database: {
      ...defaultConfig.database,
      poolMin: 5,
      poolMax: 20,
      ssl: true,
    },
    cache: {
      ...defaultConfig.cache,
      ttl: 600,
      maxItems: 10000,
      strategy: 'redis',
    },
    features: {
      ...defaultConfig.features,
      analytics: true,
    },
    security: {
      ...defaultConfig.security,
      corsOrigins: ['https://www.example.com', 'https://admin.example.com'],
      rateLimitMax: 50,
    },
    logging: {
      level: 'warn',
      format: 'json',
      destination: 'stdout',
    },
  },
};

// 設定のマージとエクスポート
function deepMerge<T extends Record<string, any>>(target: T, source: Partial<T>): T {
  const result = { ...target };
  for (const key in source) {
    if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
      result[key] = deepMerge(
        result[key] as Record<string, any>,
        source[key] as Record<string, any>
      ) as T[Extract<keyof T, string>];
    } else if (source[key] !== undefined) {
      result[key] = source[key] as T[Extract<keyof T, string>];
    }
  }
  return result;
}

const currentEnv = process.env.NODE_ENV || 'development';
const override = envOverrides[currentEnv] || {};

export const config: AppConfig = deepMerge(defaultConfig, override);

// 設定の凍結（実行時の書き換え防止）
Object.freeze(config);
Object.keys(config).forEach(key => {
  if (typeof (config as any)[key] === 'object') {
    Object.freeze((config as any)[key]);
  }
});
```

### 3.2 設定のDI（依存性注入）パターン

```typescript
// ============================================
// 設定をDIコンテナで管理するパターン
// ============================================

// config/types.ts
export interface DatabaseConfig {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
  ssl: boolean;
  poolSize: number;
}

export interface CacheConfig {
  driver: 'memory' | 'redis' | 'memcached';
  ttl: number;
  prefix: string;
  redis?: {
    url: string;
    password?: string;
  };
}

export interface EmailConfig {
  provider: 'sendgrid' | 'ses' | 'smtp';
  from: string;
  replyTo?: string;
  apiKey?: string;
  smtp?: {
    host: string;
    port: number;
    secure: boolean;
  };
}

// config/container.ts
import { DatabaseConfig, CacheConfig, EmailConfig } from './types';

class ConfigContainer {
  private configs = new Map<string, unknown>();

  register<T>(key: string, config: T): void {
    this.configs.set(key, Object.freeze(config));
  }

  get<T>(key: string): T {
    const config = this.configs.get(key);
    if (!config) {
      throw new Error(`設定 "${key}" が登録されていません`);
    }
    return config as T;
  }

  has(key: string): boolean {
    return this.configs.has(key);
  }
}

// シングルトンインスタンス
export const configContainer = new ConfigContainer();

// 初期化
export function initializeConfig(): void {
  configContainer.register<DatabaseConfig>('database', {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432'),
    database: process.env.DB_NAME || 'myapp',
    username: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || '',
    ssl: process.env.DB_SSL === 'true',
    poolSize: parseInt(process.env.DB_POOL_SIZE || '10'),
  });

  configContainer.register<CacheConfig>('cache', {
    driver: (process.env.CACHE_DRIVER as CacheConfig['driver']) || 'memory',
    ttl: parseInt(process.env.CACHE_TTL || '300'),
    prefix: process.env.CACHE_PREFIX || 'myapp:',
    redis: process.env.REDIS_URL
      ? { url: process.env.REDIS_URL, password: process.env.REDIS_PASSWORD }
      : undefined,
  });

  configContainer.register<EmailConfig>('email', {
    provider: (process.env.EMAIL_PROVIDER as EmailConfig['provider']) || 'sendgrid',
    from: process.env.EMAIL_FROM || 'noreply@example.com',
    replyTo: process.env.EMAIL_REPLY_TO,
    apiKey: process.env.SENDGRID_API_KEY,
  });
}

// 使用例
// import { configContainer, initializeConfig } from './config/container';
// initializeConfig();
// const dbConfig = configContainer.get<DatabaseConfig>('database');
```

---

## 4. Feature Flags

### 4.1 Feature Flags の概要と種類

Feature Flags（機能フラグ）は、コードの変更なしに機能の有効/無効を切り替える仕組みである。デプロイとリリースを分離し、段階的なロールアウトやA/Bテストを実現する。

```
Feature Flags の種類:

  ┌───────────────────┬──────────────────────────────────────────────┐
  │ 種類               │ 説明                                        │
  ├───────────────────┼──────────────────────────────────────────────┤
  │ Release Flag      │ 未完成の機能を隠すためのフラグ                │
  │                   │ 開発完了後に削除する（短命）                   │
  │                   │ 例: 新しいダッシュボード UI                    │
  ├───────────────────┼──────────────────────────────────────────────┤
  │ Experiment Flag   │ A/Bテストやパーセンテージロールアウト用         │
  │                   │ 実験完了後に削除する（中期）                   │
  │                   │ 例: 新しいチェックアウトフロー                 │
  ├───────────────────┼──────────────────────────────────────────────┤
  │ Ops Flag          │ 運用上のトグル（メンテナンスモードなど）        │
  │                   │ 長期的に維持する                              │
  │                   │ 例: 書き込み操作の一時停止                    │
  ├───────────────────┼──────────────────────────────────────────────┤
  │ Permission Flag   │ ユーザーの権限やプランに基づく機能制御          │
  │                   │ 永続的に維持する                              │
  │                   │ 例: プレミアムプラン限定機能                   │
  └───────────────────┴──────────────────────────────────────────────┘

Feature Flags のライフサイクル:
  作成 → テスト → 段階的有効化 → 全体有効化 → フラグ削除

  注意: 不要になったフラグは必ず削除する（技術的負債の温床）
```

### 4.2 環境変数ベースの Feature Flags 実装

```typescript
// ============================================
// lib/feature-flags.ts
// 環境変数ベースのシンプルな Feature Flags
// ============================================

// Feature Flag の定義
const FEATURE_FLAGS = {
  // Release Flags
  newDashboard: {
    envVar: 'NEXT_PUBLIC_FF_NEW_DASHBOARD',
    description: '新しいダッシュボード UI',
    defaultValue: false,
  },
  newCheckout: {
    envVar: 'NEXT_PUBLIC_FF_NEW_CHECKOUT',
    description: '新しいチェックアウトフロー',
    defaultValue: false,
  },

  // Ops Flags
  maintenanceMode: {
    envVar: 'NEXT_PUBLIC_FF_MAINTENANCE',
    description: 'メンテナンスモード',
    defaultValue: false,
  },
  readOnlyMode: {
    envVar: 'NEXT_PUBLIC_FF_READ_ONLY',
    description: '読み取り専用モード',
    defaultValue: false,
  },

  // Experiment Flags
  darkMode: {
    envVar: 'NEXT_PUBLIC_FF_DARK_MODE',
    description: 'ダークモード',
    defaultValue: false,
  },

  // Permission Flags
  betaFeatures: {
    envVar: 'NEXT_PUBLIC_FF_BETA',
    description: 'ベータ機能の表示',
    defaultValue: false,
  },
  premiumFeatures: {
    envVar: 'NEXT_PUBLIC_FF_PREMIUM',
    description: 'プレミアム機能',
    defaultValue: false,
  },
} as const;

type FeatureFlagName = keyof typeof FEATURE_FLAGS;

// Feature Flag の状態を取得
export function isFeatureEnabled(name: FeatureFlagName): boolean {
  const flag = FEATURE_FLAGS[name];
  const envValue = process.env[flag.envVar];

  if (envValue === undefined || envValue === '') {
    return flag.defaultValue;
  }

  return envValue === 'true' || envValue === '1';
}

// 全 Feature Flags の状態を取得
export function getAllFeatureFlags(): Record<FeatureFlagName, boolean> {
  const flags = {} as Record<FeatureFlagName, boolean>;

  for (const name of Object.keys(FEATURE_FLAGS) as FeatureFlagName[]) {
    flags[name] = isFeatureEnabled(name);
  }

  return flags;
}

// Feature Flag の説明を取得（管理画面用）
export function getFeatureFlagDescriptions(): Array<{
  name: string;
  description: string;
  enabled: boolean;
  envVar: string;
}> {
  return (Object.entries(FEATURE_FLAGS) as [FeatureFlagName, typeof FEATURE_FLAGS[FeatureFlagName]][])
    .map(([name, flag]) => ({
      name,
      description: flag.description,
      enabled: isFeatureEnabled(name as FeatureFlagName),
      envVar: flag.envVar,
    }));
}
```

### 4.3 React コンポーネントとしての Feature Flag

```tsx
// ============================================
// components/FeatureFlag.tsx
// React コンポーネントとして Feature Flag を使う
// ============================================
import React, { createContext, useContext, ReactNode } from 'react';
import { getAllFeatureFlags, isFeatureEnabled } from '@/lib/feature-flags';

type FeatureFlagName = Parameters<typeof isFeatureEnabled>[0];

// ---- Context API による Feature Flags の提供 ----
interface FeatureFlagContextType {
  flags: Record<string, boolean>;
  isEnabled: (name: FeatureFlagName) => boolean;
}

const FeatureFlagContext = createContext<FeatureFlagContextType>({
  flags: {},
  isEnabled: () => false,
});

export function FeatureFlagProvider({ children }: { children: ReactNode }) {
  const flags = getAllFeatureFlags();

  const contextValue: FeatureFlagContextType = {
    flags,
    isEnabled: (name: FeatureFlagName) => flags[name] ?? false,
  };

  return (
    <FeatureFlagContext.Provider value={contextValue}>
      {children}
    </FeatureFlagContext.Provider>
  );
}

// ---- Hook ----
export function useFeatureFlag(name: FeatureFlagName): boolean {
  const context = useContext(FeatureFlagContext);
  return context.isEnabled(name);
}

export function useFeatureFlags() {
  return useContext(FeatureFlagContext);
}

// ---- 宣言的コンポーネント ----
interface FeatureFlagProps {
  name: FeatureFlagName;
  children: ReactNode;
  fallback?: ReactNode;
}

export function FeatureFlag({ name, children, fallback = null }: FeatureFlagProps) {
  const isEnabled = useFeatureFlag(name);
  return isEnabled ? <>{children}</> : <>{fallback}</>;
}

// ---- 使用例 ----
// function App() {
//   return (
//     <FeatureFlagProvider>
//       <Layout>
//         <FeatureFlag
//           name="newDashboard"
//           fallback={<OldDashboard />}
//         >
//           <NewDashboard />
//         </FeatureFlag>
//
//         <FeatureFlag name="darkMode">
//           <DarkModeToggle />
//         </FeatureFlag>
//       </Layout>
//     </FeatureFlagProvider>
//   );
// }

// ---- Hook の使用例 ----
// function NavigationMenu() {
//   const showBeta = useFeatureFlag('betaFeatures');
//   const showPremium = useFeatureFlag('premiumFeatures');
//
//   return (
//     <nav>
//       <Link href="/">Home</Link>
//       <Link href="/dashboard">Dashboard</Link>
//       {showBeta && <Link href="/beta">Beta Features</Link>}
//       {showPremium && <Link href="/premium">Premium</Link>}
//     </nav>
//   );
// }
```

### 4.4 サービスベースの Feature Flags（段階的ロールアウト）

```typescript
// ============================================
// lib/feature-flags-service.ts
// 外部サービスを利用した高度な Feature Flags
// ============================================

// Feature Flag サービスの抽象インターフェース
interface FeatureFlagService {
  isEnabled(flagName: string, context?: EvaluationContext): Promise<boolean>;
  getVariant(flagName: string, context?: EvaluationContext): Promise<string | null>;
  getAllFlags(context?: EvaluationContext): Promise<Record<string, boolean>>;
}

interface EvaluationContext {
  userId?: string;
  email?: string;
  country?: string;
  plan?: 'free' | 'pro' | 'enterprise';
  percentile?: number; // 0-100 のハッシュ値
  attributes?: Record<string, string | number | boolean>;
}

// パーセンテージベースのロールアウト実装
class PercentageRollout {
  // ユーザーIDからハッシュ値を計算（0-100）
  static calculatePercentile(userId: string, flagName: string): number {
    const input = `${userId}:${flagName}`;
    let hash = 0;
    for (let i = 0; i < input.length; i++) {
      const char = input.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // 32bit integer に変換
    }
    return Math.abs(hash) % 100;
  }

  // パーセンテージでの有効判定
  static isEnabledForUser(
    userId: string,
    flagName: string,
    percentage: number
  ): boolean {
    const userPercentile = this.calculatePercentile(userId, flagName);
    return userPercentile < percentage;
  }
}

// ローカル実装（外部サービスなし）
class LocalFeatureFlagService implements FeatureFlagService {
  private flags: Map<string, FlagConfig> = new Map();

  constructor(config: Record<string, FlagConfig>) {
    for (const [name, flagConfig] of Object.entries(config)) {
      this.flags.set(name, flagConfig);
    }
  }

  async isEnabled(
    flagName: string,
    context?: EvaluationContext
  ): Promise<boolean> {
    const flag = this.flags.get(flagName);
    if (!flag) return false;

    // グローバルに無効の場合
    if (!flag.enabled) return false;

    // コンテキストなしの場合はグローバル設定を返す
    if (!context) return flag.enabled;

    // ユーザーセグメントのチェック
    if (flag.allowedUsers?.includes(context.userId || '')) return true;
    if (flag.allowedEmails?.includes(context.email || '')) return true;
    if (flag.blockedUsers?.includes(context.userId || '')) return false;

    // プラン制限のチェック
    if (flag.requiredPlans && context.plan) {
      if (!flag.requiredPlans.includes(context.plan)) return false;
    }

    // パーセンテージロールアウト
    if (flag.rolloutPercentage !== undefined && context.userId) {
      return PercentageRollout.isEnabledForUser(
        context.userId,
        flagName,
        flag.rolloutPercentage
      );
    }

    return flag.enabled;
  }

  async getVariant(
    flagName: string,
    context?: EvaluationContext
  ): Promise<string | null> {
    const flag = this.flags.get(flagName);
    if (!flag?.variants || !context?.userId) return null;

    const percentile = PercentageRollout.calculatePercentile(
      context.userId,
      flagName
    );

    let cumulative = 0;
    for (const variant of flag.variants) {
      cumulative += variant.weight;
      if (percentile < cumulative) return variant.name;
    }

    return null;
  }

  async getAllFlags(
    context?: EvaluationContext
  ): Promise<Record<string, boolean>> {
    const result: Record<string, boolean> = {};
    for (const [name] of this.flags) {
      result[name] = await this.isEnabled(name, context);
    }
    return result;
  }
}

interface FlagConfig {
  enabled: boolean;
  description?: string;
  rolloutPercentage?: number; // 0-100
  allowedUsers?: string[];
  allowedEmails?: string[];
  blockedUsers?: string[];
  requiredPlans?: Array<'free' | 'pro' | 'enterprise'>;
  variants?: Array<{ name: string; weight: number }>;
}

// サービスのインスタンス作成
export const featureFlagService = new LocalFeatureFlagService({
  'new-checkout': {
    enabled: true,
    description: '新しいチェックアウトフロー',
    rolloutPercentage: 25, // 25% のユーザーに有効
  },
  'premium-analytics': {
    enabled: true,
    description: 'プレミアム分析ダッシュボード',
    requiredPlans: ['pro', 'enterprise'],
  },
  'ab-test-pricing': {
    enabled: true,
    description: '料金ページの A/B テスト',
    variants: [
      { name: 'control', weight: 50 },
      { name: 'variant-a', weight: 25 },
      { name: 'variant-b', weight: 25 },
    ],
  },
  'beta-ai-features': {
    enabled: true,
    description: 'AI 機能のベータテスト',
    allowedEmails: ['beta-tester@example.com'],
    rolloutPercentage: 5,
  },
});

// 使用例
// const isNewCheckout = await featureFlagService.isEnabled('new-checkout', {
//   userId: currentUser.id,
//   plan: currentUser.plan,
// });
```

### 4.5 Feature Flags サービスの比較

```
主要な Feature Flags サービスの比較:

  ┌───────────────┬─────────┬──────────────┬──────────────┬───────────┐
  │ サービス       │ 料金     │ セグメント    │ A/Bテスト    │ OSS       │
  ├───────────────┼─────────┼──────────────┼──────────────┼───────────┤
  │ LaunchDarkly  │ 有料     │ 高度         │ 対応         │ ×         │
  │ Unleash       │ 無料+有料│ 中程度       │ 対応         │ ○         │
  │ Flagsmith     │ 無料+有料│ 中程度       │ 対応         │ ○         │
  │ PostHog       │ 無料+有料│ 高度         │ 対応         │ ○         │
  │ ConfigCat     │ 無料+有料│ 中程度       │ 限定的       │ ×         │
  │ Split.io      │ 有料     │ 高度         │ 対応         │ ×         │
  │ 環境変数ベース │ 無料     │ なし         │ なし         │ -         │
  │ Vercel Edge   │ 無料+有料│ 地理・デバイス│ 対応         │ ×         │
  │ Config        │          │              │              │           │
  └───────────────┴─────────┴──────────────┴──────────────┴───────────┘

選択の指針:
  - 小規模プロジェクト → 環境変数ベース or Unleash（セルフホスト）
  - 中規模プロジェクト → Flagsmith or PostHog
  - 大規模プロジェクト → LaunchDarkly or Split.io
  - A/Bテスト重視     → PostHog or LaunchDarkly
  - OSS 重視          → Unleash or Flagsmith
```

---

## 5. シークレット管理

### 5.1 シークレットの分類と管理方針

```
シークレットの分類:

  ┌─────────────────────┬───────────────────────┬──────────────────┐
  │ 分類                 │ 例                     │ 管理方法          │
  ├─────────────────────┼───────────────────────┼──────────────────┤
  │ API キー             │ Stripe Secret Key     │ 環境変数 or Vault │
  │ データベース認証情報   │ DB パスワード、接続URI │ Secrets Manager  │
  │ 暗号化キー           │ JWT Secret, AES Key   │ KMS or Vault     │
  │ OAuth クレデンシャル  │ Client Secret         │ Secrets Manager  │
  │ SSH キー             │ デプロイキー            │ CI/CD シークレット│
  │ TLS 証明書           │ SSL 証明書・秘密鍵     │ Certificate Mgr  │
  │ Webhook シークレット  │ Stripe Webhook Secret │ 環境変数          │
  └─────────────────────┴───────────────────────┴──────────────────┘

シークレット管理の原則:
  ① 最小権限の原則: 必要な最小限のアクセス権のみ付与
  ② ローテーション: 定期的にシークレットを更新する
  ③ 暗号化: 保存時も転送時も暗号化する
  ④ 監査: シークレットへのアクセスを記録・監視する
  ⑤ 分離: 環境ごとに異なるシークレットを使用する
```

### 5.2 各プラットフォームでのシークレット管理

```
シークレットの管理方法:

  ① 環境変数（基本）:
     → Vercel: Project Settings > Environment Variables
        - Preview / Production / Development で分離可能
        - Sensitive フラグでログ出力を防止
     → AWS: Systems Manager Parameter Store
        - SecureString 型で暗号化保存
        - IAM ポリシーでアクセス制御
     → .env.local（ローカル開発のみ）
        - 絶対に Git にコミットしない

  ② シークレットマネージャー:
     → AWS Secrets Manager
        - 自動ローテーション対応
        - バージョン管理
        - クロスリージョンレプリケーション
     → Google Cloud Secret Manager
        - IAM による細かいアクセス制御
        - 自動レプリケーション
     → HashiCorp Vault
        - 動的シークレット生成
        - リース（有効期限）管理
        - 詳細な監査ログ
     → Azure Key Vault
        - HSM（ハードウェアセキュリティモジュール）対応
        - 証明書管理も統合

  ③ .env ファイルの暗号化:
     → dotenv-vault
        - .env ファイルを暗号化してコミット可能
        - チーム間で安全に共有
     → sops（Mozilla）
        - AWS KMS / GCP KMS / Azure Key Vault と統合
        - 部分暗号化（キーは平文、値のみ暗号化）
     → git-crypt
        - Git リポジトリ内のファイルを透過的に暗号化
        - GPG キーベースのアクセス制御
```

### 5.3 AWS Secrets Manager の実装例

```typescript
// ============================================
// lib/secrets.ts
// AWS Secrets Manager を使ったシークレット取得
// ============================================
import {
  SecretsManagerClient,
  GetSecretValueCommand,
} from '@aws-sdk/client-secrets-manager';

const client = new SecretsManagerClient({
  region: process.env.AWS_REGION || 'ap-northeast-1',
});

// シークレットのキャッシュ（メモリ内）
const secretCache = new Map<string, { value: string; expiresAt: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5分

export async function getSecret(secretName: string): Promise<string> {
  // キャッシュを確認
  const cached = secretCache.get(secretName);
  if (cached && cached.expiresAt > Date.now()) {
    return cached.value;
  }

  try {
    const command = new GetSecretValueCommand({
      SecretId: secretName,
    });

    const response = await client.send(command);
    const secretValue = response.SecretString;

    if (!secretValue) {
      throw new Error(`Secret "${secretName}" has no value`);
    }

    // キャッシュに保存
    secretCache.set(secretName, {
      value: secretValue,
      expiresAt: Date.now() + CACHE_TTL,
    });

    return secretValue;
  } catch (error) {
    console.error(`Failed to retrieve secret "${secretName}":`, error);
    throw error;
  }
}

// JSON 形式のシークレットをパース
export async function getSecretJson<T>(secretName: string): Promise<T> {
  const secretString = await getSecret(secretName);
  return JSON.parse(secretString) as T;
}

// 使用例: データベース接続情報をシークレットから取得
interface DbCredentials {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
}

export async function getDatabaseCredentials(): Promise<DbCredentials> {
  // 開発環境では環境変数から、本番ではシークレットマネージャーから取得
  if (process.env.NODE_ENV === 'development') {
    return {
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT || '5432'),
      database: process.env.DB_NAME || 'myapp_dev',
      username: process.env.DB_USER || 'postgres',
      password: process.env.DB_PASSWORD || 'password',
    };
  }

  return getSecretJson<DbCredentials>('myapp/production/database');
}
```

### 5.4 シークレット漏洩の防止策

```
やってはいけないこと:
  ✗ シークレットをコードにハードコード
  ✗ .env.local を Git にコミット
  ✗ NEXT_PUBLIC_ にシークレットを入れる
  ✗ ログにシークレットを出力
  ✗ エラーメッセージにシークレットを含める
  ✗ シークレットをURL のクエリパラメータに含める
  ✗ フロントエンドの JavaScript にシークレットを埋め込む
  ✗ コメントやドキュメントに実際のシークレットを記載

.gitignore の設定:
  .env.local
  .env.*.local
  .env.development.local
  .env.production.local
  *.pem
  *.key
  *.p12
  credentials.json
  service-account.json
```

```typescript
// ============================================
// pre-commit hook でシークレット漏洩を検出
// .husky/pre-commit
// ============================================

// package.json に追加するスクリプト
// {
//   "scripts": {
//     "check-secrets": "ts-node scripts/check-secrets.ts"
//   },
//   "lint-staged": {
//     "**/*": "ts-node scripts/check-secrets.ts"
//   }
// }

// scripts/check-secrets.ts
import { execSync } from 'child_process';

// 危険なパターンの定義
const SECRET_PATTERNS = [
  // AWS
  { pattern: /AKIA[0-9A-Z]{16}/g, description: 'AWS Access Key ID' },
  { pattern: /(?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY)\s*=\s*[\w/+=]{40}/g,
    description: 'AWS Secret Access Key' },

  // Stripe
  { pattern: /sk_live_[a-zA-Z0-9]{24,}/g, description: 'Stripe Live Secret Key' },
  { pattern: /rk_live_[a-zA-Z0-9]{24,}/g, description: 'Stripe Live Restricted Key' },

  // GitHub
  { pattern: /ghp_[a-zA-Z0-9]{36}/g, description: 'GitHub Personal Access Token' },
  { pattern: /ghs_[a-zA-Z0-9]{36}/g, description: 'GitHub App Installation Token' },

  // Google
  { pattern: /AIza[0-9A-Za-z\-_]{35}/g, description: 'Google API Key' },

  // 汎用
  { pattern: /-----BEGIN (?:RSA |EC )?PRIVATE KEY-----/g, description: 'Private Key' },
  { pattern: /password\s*=\s*['"][^'"]{8,}['"]/gi, description: 'Hardcoded Password' },
];

function checkForSecrets(filePaths: string[]): void {
  const violations: Array<{ file: string; line: number; description: string }> = [];

  for (const filePath of filePaths) {
    // バイナリファイルはスキップ
    if (/\.(png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot)$/.test(filePath)) {
      continue;
    }

    try {
      const content = require('fs').readFileSync(filePath, 'utf-8');
      const lines = content.split('\n');

      for (let lineNum = 0; lineNum < lines.length; lineNum++) {
        const line = lines[lineNum];

        for (const { pattern, description } of SECRET_PATTERNS) {
          // パターンをリセット（global フラグのため）
          pattern.lastIndex = 0;
          if (pattern.test(line)) {
            violations.push({
              file: filePath,
              line: lineNum + 1,
              description,
            });
          }
        }
      }
    } catch (error) {
      // ファイル読み込みエラーは無視
    }
  }

  if (violations.length > 0) {
    console.error('\nシークレット漏洩の可能性を検出しました:');
    for (const v of violations) {
      console.error(`  ${v.file}:${v.line} - ${v.description}`);
    }
    console.error('\nシークレットを削除してから再度コミットしてください。');
    process.exit(1);
  }

  console.log('シークレットチェック: 問題は検出されませんでした。');
}

// ステージングされたファイルを取得
const stagedFiles = execSync('git diff --cached --name-only --diff-filter=ACM')
  .toString()
  .trim()
  .split('\n')
  .filter(Boolean);

checkForSecrets(stagedFiles);
```

---

## 6. CI/CD パイプラインでの環境変数管理

### 6.1 GitHub Actions での環境変数設定

```yaml
# ============================================
# .github/workflows/deploy.yml
# GitHub Actions での環境変数管理
# ============================================
name: Deploy

on:
  push:
    branches: [main, staging]

# 環境変数の設定方法:
# 1. Repository secrets: Settings > Secrets and variables > Actions
# 2. Environment secrets: Settings > Environments > [env] > Secrets
# 3. Organization secrets: Organization Settings > Secrets

jobs:
  deploy:
    runs-on: ubuntu-latest

    # 環境を指定（GitHub Environments）
    environment:
      name: ${{ github.ref == 'refs/heads/main' && 'production' || 'staging' }}
      url: ${{ steps.deploy.outputs.url }}

    # ジョブレベルの環境変数
    env:
      NODE_ENV: production
      NEXT_TELEMETRY_DISABLED: 1

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run environment check
        run: npm run check-env
        env:
          # Repository secrets から注入
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          JWT_SECRET: ${{ secrets.JWT_SECRET }}
          STRIPE_SECRET_KEY: ${{ secrets.STRIPE_SECRET_KEY }}
          STRIPE_WEBHOOK_SECRET: ${{ secrets.STRIPE_WEBHOOK_SECRET }}
          SENDGRID_API_KEY: ${{ secrets.SENDGRID_API_KEY }}
          # Environment secrets から注入（環境ごとに異なる値）
          NEXT_PUBLIC_API_URL: ${{ vars.NEXT_PUBLIC_API_URL }}
          NEXT_PUBLIC_APP_URL: ${{ vars.NEXT_PUBLIC_APP_URL }}

      - name: Build
        run: npm run build
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          NEXT_PUBLIC_API_URL: ${{ vars.NEXT_PUBLIC_API_URL }}
          NEXT_PUBLIC_APP_URL: ${{ vars.NEXT_PUBLIC_APP_URL }}
          NEXT_PUBLIC_GA_ID: ${{ vars.NEXT_PUBLIC_GA_ID }}
          NEXT_PUBLIC_SENTRY_DSN: ${{ vars.NEXT_PUBLIC_SENTRY_DSN }}
          SENTRY_AUTH_TOKEN: ${{ secrets.SENTRY_AUTH_TOKEN }}

      - name: Deploy to Vercel
        id: deploy
        run: |
          npx vercel deploy --prod --token=${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}

      - name: Run smoke tests
        run: npm run test:smoke
        env:
          TEST_URL: ${{ steps.deploy.outputs.url }}

  # 環境変数の設定確認ジョブ
  verify-config:
    runs-on: ubuntu-latest
    needs: deploy
    steps:
      - name: Verify deployment config
        run: |
          RESPONSE=$(curl -s ${{ needs.deploy.outputs.url }}/api/health)
          echo "Health check response: $RESPONSE"
          if echo "$RESPONSE" | jq -e '.status == "ok"' > /dev/null 2>&1; then
            echo "Deployment verification passed"
          else
            echo "Deployment verification failed"
            exit 1
          fi
```

### 6.2 Vercel での環境変数設定

```typescript
// ============================================
// vercel.json - Vercel プロジェクト設定
// ============================================
// {
//   "env": {
//     "CUSTOM_VAR": "value"
//   },
//   "build": {
//     "env": {
//       "BUILD_VAR": "build-value"
//     }
//   }
// }

// Vercel CLI での環境変数設定
// vercel env add DATABASE_URL production
// vercel env add DATABASE_URL preview
// vercel env add DATABASE_URL development

// 環境変数の一覧取得
// vercel env ls

// 環境変数の削除
// vercel env rm DATABASE_URL production

// ============================================
// Vercel の環境変数のスコープ
// ============================================
// Production:   本番デプロイ時のみ使用
// Preview:      プレビューデプロイ（PR ブランチ）で使用
// Development:  vercel dev 実行時に使用
//
// Sensitive:    ログやビルド出力に表示されない
// Plain:        通常の環境変数

// ============================================
// vercel.json で Preview 環境のブランチ固有設定
// ============================================
// {
//   "git": {
//     "deploymentEnabled": {
//       "feature/*": true,
//       "fix/*": true,
//       "main": true
//     }
//   }
// }
```

### 6.3 Docker での環境変数管理

```dockerfile
# ============================================
# Dockerfile - マルチステージビルドでの環境変数
# ============================================

# ---- ビルドステージ ----
FROM node:20-alpine AS builder

WORKDIR /app

# ビルド時の環境変数（ARG）
# docker build --build-arg NEXT_PUBLIC_API_URL=https://api.example.com .
ARG NEXT_PUBLIC_API_URL
ARG NEXT_PUBLIC_APP_URL
ARG NEXT_PUBLIC_GA_ID

# ARG を ENV に変換（ビルドプロセスで使用）
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_APP_URL=$NEXT_PUBLIC_APP_URL
ENV NEXT_PUBLIC_GA_ID=$NEXT_PUBLIC_GA_ID

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---- 実行ステージ ----
FROM node:20-alpine AS runner

WORKDIR /app

# 実行時の環境変数はここでは設定しない
# docker run -e DATABASE_URL=... で注入する
ENV NODE_ENV=production

# セキュリティ: 非 root ユーザーで実行
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

CMD ["node", "server.js"]
```

```yaml
# ============================================
# docker-compose.yml - 開発環境の構成
# ============================================
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
      args:
        - NEXT_PUBLIC_API_URL=http://localhost:3001
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgresql://postgres:password@db:5432/myapp_dev
      - REDIS_URL=redis://redis:6379
    env_file:
      - .env                    # 共通設定
      - .env.development        # 開発環境設定
      - .env.local              # ローカルオーバーライド（.gitignore）
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - .:/app
      - /app/node_modules

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: myapp_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

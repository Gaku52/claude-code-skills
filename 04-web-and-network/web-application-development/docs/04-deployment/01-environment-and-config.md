# 環境設定

> 環境設定はアプリケーションの安全な運用の基盤。環境変数の管理、Feature Flags、設定の階層化、シークレット管理まで、本番環境で安全に設定を管理するベストプラクティスを習得する。

## この章で学ぶこと

- [ ] 環境変数の設計と安全な管理を理解する
- [ ] Feature Flagsの実装と運用を把握する
- [ ] シークレット管理のベストプラクティスを学ぶ

---

## 1. 環境変数の設計

```
環境の階層:
  development → staging → production

  .env                  ← デフォルト（全環境共通）
  .env.local            ← ローカルのオーバーライド（.gitignore）
  .env.development      ← 開発環境
  .env.staging          ← ステージング
  .env.production       ← 本番環境

Next.js の環境変数:
  NEXT_PUBLIC_*  → クライアント + サーバーで使用可能（公開）
  それ以外       → サーバーのみ（機密情報）

  // 使い分け
  NEXT_PUBLIC_API_URL=https://api.example.com  ← クライアントで必要
  DATABASE_URL=postgresql://...                ← サーバーのみ
  STRIPE_SECRET_KEY=sk_live_...                ← サーバーのみ

命名規則:
  ✓ SCREAMING_SNAKE_CASE
  ✓ プレフィックスで分類: DB_, AWS_, NEXT_PUBLIC_
  ✓ ブール値: ENABLE_*, IS_*, HAS_*
  ✗ 汎用的な名前: KEY, SECRET, URL
```

```typescript
// 型安全な環境変数（Zod）
import { z } from 'zod';

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']),
  DATABASE_URL: z.string().url(),
  NEXT_PUBLIC_API_URL: z.string().url(),
  STRIPE_SECRET_KEY: z.string().startsWith('sk_'),
  REDIS_URL: z.string().url().optional(),
  ENABLE_ANALYTICS: z.coerce.boolean().default(false),
});

// 起動時にバリデーション
export const env = envSchema.parse(process.env);

// 使用（型安全）
console.log(env.DATABASE_URL);       // string
console.log(env.ENABLE_ANALYTICS);   // boolean
```

---

## 2. Feature Flags

```typescript
// 環境変数ベースの Feature Flags
const features = {
  newDashboard: process.env.NEXT_PUBLIC_FF_NEW_DASHBOARD === 'true',
  darkMode: process.env.NEXT_PUBLIC_FF_DARK_MODE === 'true',
  betaFeatures: process.env.NEXT_PUBLIC_FF_BETA === 'true',
};

function FeatureFlag({ name, children, fallback = null }: {
  name: keyof typeof features;
  children: React.ReactNode;
  fallback?: React.ReactNode;
}) {
  return features[name] ? <>{children}</> : <>{fallback}</>;
}

// 使用
<FeatureFlag name="newDashboard" fallback={<OldDashboard />}>
  <NewDashboard />
</FeatureFlag>

// サービスベース Feature Flags（段階的ロールアウト）
// → LaunchDarkly, Unleash, Flagsmith
// → ユーザーセグメント別の有効化
// → A/Bテスト
// → パーセンテージロールアウト
```

---

## 3. シークレット管理

```
シークレットの管理方法:

  ① 環境変数（基本）:
     → Vercel: Project Settings > Environment Variables
     → AWS: Systems Manager Parameter Store
     → .env.local（ローカル開発のみ）

  ② シークレットマネージャー:
     → AWS Secrets Manager
     → HashiCorp Vault
     → 1Password（開発者向け）

  ③ .env ファイルの暗号化:
     → dotenv-vault
     → sops（Mozilla）

やってはいけないこと:
  ✗ シークレットをコードにハードコード
  ✗ .env.local を Git にコミット
  ✗ NEXT_PUBLIC_ にシークレットを入れる
  ✗ ログにシークレットを出力
  ✗ エラーメッセージにシークレットを含める

.gitignore:
  .env.local
  .env.*.local
  .env.development.local
  .env.production.local
```

---

## 4. 設定の分離

```typescript
// 環境別設定ファイル
// config/index.ts
const configs = {
  development: {
    api: { baseUrl: 'http://localhost:3001', timeout: 30000 },
    features: { analytics: false, debugMode: true },
    cache: { ttl: 0 },
  },
  staging: {
    api: { baseUrl: 'https://staging-api.example.com', timeout: 15000 },
    features: { analytics: true, debugMode: true },
    cache: { ttl: 60 },
  },
  production: {
    api: { baseUrl: 'https://api.example.com', timeout: 10000 },
    features: { analytics: true, debugMode: false },
    cache: { ttl: 300 },
  },
} as const;

type Env = keyof typeof configs;
const currentEnv = (process.env.NODE_ENV || 'development') as Env;

export const config = configs[currentEnv];
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 環境変数 | NEXT_PUBLIC_ = クライアント公開 |
| 型安全 | Zod で起動時バリデーション |
| Feature Flags | 段階的ロールアウト |
| シークレット | .env.local は Git 管理外 |

---

## 次に読むべきガイド
→ [[02-performance-optimization.md]] — パフォーマンス

---

## 参考文献
1. Next.js. "Environment Variables." nextjs.org/docs, 2024.
2. Vercel. "Environment Variables." vercel.com/docs, 2024.

# デプロイ先

> デプロイ先の選択はアプリの性能・コスト・運用に直結する。Vercel、Cloudflare、AWS、Docker、各プラットフォームの特徴と選定基準を理解し、プロジェクトに最適なデプロイ戦略を選択する。

## この章で学ぶこと

- [ ] 主要なデプロイプラットフォームの比較を理解する
- [ ] プロジェクト要件に基づく選定基準を把握する
- [ ] Docker化とセルフホスティングのパターンを学ぶ

---

## 1. プラットフォーム比較

```
                Vercel     Cloudflare   AWS         Docker
─────────────────────────────────────────────────────
対象           Next.js     全般         全般        全般
サーバーレス    ○          ○(Workers)   ○(Lambda)   ×
エッジ         ○          ◎            ○           ×
SSR            ○          △            ○           ○
コスト         中          安い         変動        インフラ次第
スケーリング   自動        自動         自動/手動    手動/K8s
カスタマイズ   低          中           高          最高
学習コスト     最低        低           高          中
DB統合         △          D1,KV等      RDS等       自由

選定フロー:
  Next.js + 最速デプロイ → Vercel
  エッジ重視 + 低コスト → Cloudflare Pages/Workers
  フルカスタム + AWS既存環境 → AWS（ECS, Lambda）
  完全制御 + 既存インフラ → Docker + VPS
```

---

## 2. Vercel

```
Vercel:
  → Next.js の開発元が提供するプラットフォーム
  → Git push でデプロイ（Zero Config）
  → プレビュー環境の自動作成（PR ごと）

  セットアップ:
  1. GitHub リポジトリを接続
  2. フレームワーク自動検出
  3. git push → 自動デプロイ

  機能:
  ✓ SSR / SSG / ISR の完全サポート
  ✓ Edge Functions
  ✓ Image Optimization
  ✓ Analytics（Web Vitals）
  ✓ Preview Deployments
  ✓ Serverless Functions

  制限（Hobby プラン）:
  → 関数実行: 10秒タイムアウト
  → 帯域: 100GB/月
  → ビルド: 6000分/月

  制限（Pro プラン: $20/月）:
  → 関数実行: 60秒
  → 帯域: 1TB/月
  → チーム機能

向いているケース:
  ✓ Next.js プロジェクト
  ✓ 小〜中規模チーム
  ✓ 高速なイテレーション
  ✓ プレビュー環境が重要
```

---

## 3. Cloudflare Pages / Workers

```
Cloudflare:
  → エッジコンピューティングに特化
  → 世界300+のエッジロケーション
  → 極めて低コスト

  Pages（静的サイト + SSR）:
  → SSG/SPA の配信
  → Functions でSSR対応

  Workers（エッジサーバーレス）:
  → V8 Isolates（コールドスタートなし）
  → 0ms Cold Start
  → KV, D1(SQLite), R2(S3互換), Durable Objects

  コスト:
  Free: 10万リクエスト/日、500ビルド/月
  Paid: $5/月〜（1000万リクエスト含む）

向いているケース:
  ✓ 静的サイト + API
  ✓ エッジで動くアプリ
  ✓ 低レイテンシが最重要
  ✓ コスト最適化
```

---

## 4. Docker + セルフホスト

```dockerfile
# Next.js の本番用 Dockerfile
FROM node:20-alpine AS base

# 依存関係のインストール
FROM base AS deps
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN corepack enable pnpm && pnpm install --frozen-lockfile

# ビルド
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN corepack enable pnpm && pnpm build

# 本番イメージ
FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT=3000
CMD ["node", "server.js"]
```

```javascript
// next.config.js（standalone出力）
module.exports = {
  output: 'standalone',
};
```

---

## 5. AWS デプロイパターン

```
① AWS Amplify:
  → Vercel ライクなGitベースデプロイ
  → Next.js SSR対応
  → 管理が楽

② ECS Fargate + Docker:
  → コンテナベース
  → ALB + AutoScaling
  → フルカスタム

③ Lambda + API Gateway:
  → サーバーレス
  → コスト効率
  → コールドスタート注意

④ S3 + CloudFront（SSG/SPA）:
  → 最も安い
  → SSR 不可
  → 静的サイトのみ

選定:
  SSG/SPA → S3 + CloudFront
  SSR + 管理簡単 → Amplify
  SSR + カスタム → ECS Fargate
  API のみ → Lambda + API Gateway
```

---

## まとめ

| プラットフォーム | 最適な用途 | コスト |
|--------------|---------|--------|
| Vercel | Next.js、高速イテレーション | 中 |
| Cloudflare | エッジ、低コスト | 安 |
| AWS Amplify | AWS環境、SSR | 中 |
| Docker + VPS | フルカスタム | 変動 |

---

## 次に読むべきガイド
→ [[01-environment-and-config.md]] — 環境設定

---

## 参考文献
1. Vercel. "Documentation." vercel.com/docs, 2024.
2. Cloudflare. "Pages Documentation." developers.cloudflare.com, 2024.
3. AWS. "Amplify Documentation." docs.aws.amazon.com, 2024.

# クラウドデプロイ

> AWS、Vercel、Cloudflare Workers への実践的なデプロイ手法を習得し、プロジェクト特性に応じた最適なプラットフォームを選択する

## この章で学ぶこと

1. **AWS (ECS/Lambda/S3+CloudFront) へのデプロイ** — IaC を活用した本格的なクラウドインフラ構築とデプロイ自動化
2. **Vercel/Netlify によるフロントエンドデプロイ** — Git 連携による自動デプロイとプレビュー環境の活用
3. **Cloudflare Workers によるエッジデプロイ** — エッジコンピューティングの特性を活かしたサーバーレスデプロイ

---

## 1. クラウドデプロイの全体像

```
┌──────────────────────────────────────────────────────────┐
│               クラウドデプロイの選択肢                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────┐  フルコントロール / 高い柔軟性        │
│  │   AWS / GCP     │  EC2, ECS, EKS, Lambda              │
│  │   Azure         │  複雑だが何でもできる                  │
│  └────────┬────────┘                                     │
│           │                                              │
│  ┌────────▼────────┐  フロントエンド特化 / DX 重視         │
│  │  Vercel         │  Next.js 最適化、プレビュー環境       │
│  │  Netlify        │  JAMstack、フォーム/認証内蔵          │
│  └────────┬────────┘                                     │
│           │                                              │
│  ┌────────▼────────┐  エッジ特化 / 超低レイテンシ          │
│  │  Cloudflare     │  Workers、R2、KV、D1                 │
│  │  Workers        │  V8 Isolate ベース                   │
│  └─────────────────┘                                     │
└──────────────────────────────────────────────────────────┘
```

---

## 2. AWS デプロイ — S3 + CloudFront (静的サイト)

```yaml
# GitHub Actions — S3 + CloudFront デプロイ
name: Deploy to AWS S3 + CloudFront

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - name: Install and Build
        run: |
          npm ci
          npm run build

      - name: Configure AWS Credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions-deploy
          aws-region: ap-northeast-1

      - name: Deploy to S3
        run: |
          aws s3 sync dist/ s3://my-app-bucket \
            --delete \
            --cache-control "public, max-age=31536000, immutable" \
            --exclude "index.html"

          # index.html はキャッシュしない
          aws s3 cp dist/index.html s3://my-app-bucket/index.html \
            --cache-control "no-cache, no-store, must-revalidate"

      - name: Invalidate CloudFront Cache
        run: |
          aws cloudfront create-invalidation \
            --distribution-id ${{ secrets.CF_DISTRIBUTION_ID }} \
            --paths "/index.html" "/sw.js"
```

---

## 3. AWS Lambda デプロイ (SAM)

```yaml
# template.yaml — AWS SAM テンプレート
AWSTemplateFormatVersion: "2010-09-09"
Transform: AWS::Serverless-2016-10-31
Description: API Backend on Lambda

Globals:
  Function:
    Timeout: 30
    Runtime: nodejs20.x
    MemorySize: 256
    Environment:
      Variables:
        NODE_ENV: production
        DB_HOST: !Ref DatabaseHost

Resources:
  ApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: dist/lambda.handler
      CodeUri: .
      Events:
        Api:
          Type: HttpApi
          Properties:
            Path: /{proxy+}
            Method: ANY
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref AppTable
      AutoPublishAlias: live
      DeploymentPreference:
        Type: Canary10Percent5Minutes  # Canary デプロイ
        Alarms:
          - !Ref ApiErrorAlarm

  AppTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: app-data
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: pk
          AttributeType: S
        - AttributeName: sk
          AttributeType: S
      KeySchema:
        - AttributeName: pk
          KeyType: HASH
        - AttributeName: sk
          KeyType: RANGE

  ApiErrorAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      MetricName: 5XXError
      Namespace: AWS/ApiGateway
      Statistic: Sum
      Period: 60
      EvaluationPeriods: 1
      Threshold: 5
      ComparisonOperator: GreaterThanThreshold
```

---

## 4. Vercel デプロイ

```json
// vercel.json — Vercel 設定
{
  "framework": "nextjs",
  "regions": ["nrt1"],
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        { "key": "Cache-Control", "value": "s-maxage=0, stale-while-revalidate=60" }
      ]
    },
    {
      "source": "/(.*\\.(?:js|css|woff2|png|jpg|svg))",
      "headers": [
        { "key": "Cache-Control", "value": "public, max-age=31536000, immutable" }
      ]
    }
  ],
  "rewrites": [
    { "source": "/api/:path*", "destination": "/api/:path*" },
    { "source": "/(.*)", "destination": "/" }
  ],
  "env": {
    "DATABASE_URL": "@database-url",
    "NEXT_PUBLIC_API_URL": "https://api.example.com"
  }
}
```

```
Vercel のデプロイフロー:

  開発者                  Vercel                     CDN
    │                      │                          │
    │── git push ──────►   │                          │
    │                      │── ビルド開始              │
    │                      │   (Next.js 自動検出)      │
    │                      │                          │
    │                      │── プレビュー URL 生成     │
    │   ◄── PR コメント ── │   (*.vercel.app)         │
    │                      │                          │
    │── PR マージ ────►    │                          │
    │                      │── 本番ビルド              │
    │                      │── Edge Network 配信 ──► │
    │                      │                          │
    │                      │   Serverless Functions   │
    │                      │   Edge Functions         │
    │                      │   ISR / SSG              │
    │                      │                          │
```

---

## 5. Cloudflare Workers デプロイ

```typescript
// src/worker.ts — Cloudflare Worker
export interface Env {
  KV_STORE: KVNamespace;
  DB: D1Database;
  R2_BUCKET: R2Bucket;
}

export default {
  async fetch(
    request: Request,
    env: Env,
    ctx: ExecutionContext
  ): Promise<Response> {
    const url = new URL(request.url);

    // ルーティング
    if (url.pathname.startsWith('/api/')) {
      return handleApi(request, env, ctx);
    }

    // 静的アセットは R2 から配信
    const asset = await env.R2_BUCKET.get(url.pathname.slice(1));
    if (asset) {
      const headers = new Headers();
      headers.set('Content-Type', asset.httpMetadata?.contentType ?? 'application/octet-stream');
      headers.set('Cache-Control', 'public, max-age=86400');
      return new Response(asset.body, { headers });
    }

    return new Response('Not Found', { status: 404 });
  },
};

async function handleApi(
  request: Request,
  env: Env,
  ctx: ExecutionContext
): Promise<Response> {
  const url = new URL(request.url);

  if (url.pathname === '/api/items' && request.method === 'GET') {
    // D1 データベースクエリ
    const { results } = await env.DB
      .prepare('SELECT * FROM items ORDER BY created_at DESC LIMIT 50')
      .all();

    return Response.json(results);
  }

  if (url.pathname === '/api/items' && request.method === 'POST') {
    const body = await request.json<{ name: string; value: string }>();

    await env.DB
      .prepare('INSERT INTO items (name, value) VALUES (?, ?)')
      .bind(body.name, body.value)
      .run();

    // KV キャッシュを無効化
    ctx.waitUntil(env.KV_STORE.delete('items-cache'));

    return Response.json({ success: true }, { status: 201 });
  }

  return Response.json({ error: 'Not Found' }, { status: 404 });
}
```

```toml
# wrangler.toml — Cloudflare Workers 設定
name = "my-api"
main = "src/worker.ts"
compatibility_date = "2024-09-25"

[placement]
mode = "smart"  # スマート配置でレイテンシ最適化

[[kv_namespaces]]
binding = "KV_STORE"
id = "abc123"

[[d1_databases]]
binding = "DB"
database_name = "my-app-db"
database_id = "def456"

[[r2_buckets]]
binding = "R2_BUCKET"
bucket_name = "my-assets"

[env.production]
routes = [
  { pattern = "api.example.com/*", zone_name = "example.com" }
]
```

---

## 6. プラットフォーム比較表

| 特性 | AWS (Lambda/ECS) | Vercel | Cloudflare Workers |
|------|------------------|--------|-------------------|
| 対象 | バックエンド全般 | フロントエンド+API | エッジAPI |
| コールドスタート | 100ms〜数秒 | 数十ms | ほぼ0ms (V8 Isolate) |
| 最大実行時間 | 15分 (Lambda) | 10秒〜5分 | 30秒 (CPU 50ms/invocation) |
| メモリ上限 | 10GB (Lambda) | 1024MB | 128MB |
| ランタイム | Node.js, Python, Go等 | Node.js | JavaScript/WASM |
| DB 統合 | RDS, DynamoDB, Aurora | Vercel Postgres, KV | D1, KV, Durable Objects |
| 料金体系 | 従量課金(複雑) | 無料枠+従量 | 無料枠10万req/日 |
| 学習コスト | 高い | 低い | 中 |

| デプロイ方法比較 | Git 連携 | CLI | IaC (CDK/Terraform) |
|-----------------|---------|-----|---------------------|
| 自動化レベル | 高い | 中 | 最高 |
| 再現性 | 中 | 低い | 最高 |
| 複雑さ | 低い | 低い | 高い |
| 適用場面 | フロント / 小規模API | 開発/テスト | 本番インフラ全般 |
| ロールバック | Git revert | 手動 | 状態管理で自動 |

---

## 7. アンチパターン

### アンチパターン 1: 環境固有値のハードコード

```typescript
// 悪い例: 環境固有値をコードに埋め込む
const API_URL = "https://prod-api.example.com";
const DB_HOST = "prod-db.cluster-abc.ap-northeast-1.rds.amazonaws.com";

// 良い例: 環境変数から取得
const API_URL = process.env.API_URL;
const DB_HOST = process.env.DB_HOST;

// さらに良い例: 型安全な設定管理
import { z } from "zod";

const envSchema = z.object({
  API_URL: z.string().url(),
  DB_HOST: z.string().min(1),
  DB_PORT: z.coerce.number().default(5432),
  NODE_ENV: z.enum(["development", "staging", "production"]),
});

export const config = envSchema.parse(process.env);
```

### アンチパターン 2: キャッシュ戦略の欠如

```
[悪い例]
- 全アセットに Cache-Control なし → CDN が効かず毎回オリジンへアクセス
- index.html に長期キャッシュ → 新バージョンが配信されない
- API レスポンスにキャッシュなし → Lambda/Worker の呼び出し回数が無駄に増加

[良い例]
- 静的アセット(JS/CSS/画像): Cache-Control: public, max-age=31536000, immutable
  (ファイル名にハッシュを含める: app.a1b2c3.js)
- index.html: Cache-Control: no-cache (毎回検証)
- API: Cache-Control: s-maxage=60, stale-while-revalidate=300
  (CDN で60秒キャッシュ、バックグラウンドで300秒まで古いレスポンスを返す)
```

---

## 8. FAQ

### Q1: Vercel と AWS、どちらを選ぶべきですか？

フロントエンド（Next.js/React）が中心で、バックエンドが軽量な API Routes 程度なら Vercel が圧倒的に楽です。複雑なバックエンド処理、VPC 内のリソースアクセス、長時間バッチ処理が必要な場合は AWS を選択してください。多くのプロジェクトでは「フロントは Vercel、バックエンドは AWS」という組み合わせが実用的です。

### Q2: Cloudflare Workers の CPU 制限 (50ms) は厳しすぎませんか？

CPU 時間 50ms は「I/O 待ち時間を除いた純粋な計算時間」です。データベースクエリや外部 API 呼び出しの待ち時間は含まれません。一般的な API 処理（JSON パース、バリデーション、レスポンス構築）は数 ms で完了するため、ほとんどのユースケースでは十分です。重い計算処理が必要な場合は Workers Unbound（CPU 時間 30 秒）を検討してください。

### Q3: OIDC による AWS 認証とは何ですか？ なぜ推奨されるのですか？

GitHub Actions から AWS にアクセスする際、従来は IAM ユーザーのアクセスキーをシークレットに保存していました。OIDC（OpenID Connect）では、GitHub が発行する短命トークンを AWS が直接検証するため、長期間有効なシークレットの管理が不要になります。キーローテーションの手間がなく、漏洩リスクも低減されます。

---

## まとめ

| 項目 | 要点 |
|------|------|
| AWS S3+CloudFront | 静的サイトの定番。IaC で管理し、CloudFront で高速配信 |
| AWS Lambda (SAM) | サーバーレス API。Canary デプロイとアラーム連携が容易 |
| Vercel | Next.js 最適化。プレビュー環境と Git 連携が強力 |
| Cloudflare Workers | エッジ実行でレイテンシ最小。D1/KV/R2 のエコシステム |
| OIDC 認証 | CI/CD からのクラウド認証はシークレットキーより OIDC を推奨 |
| キャッシュ戦略 | アセットは immutable、HTML は no-cache、API は stale-while-revalidate |

---

## 次に読むべきガイド

- [00-deployment-strategies.md](./00-deployment-strategies.md) — Blue-Green、Canary などのデプロイ戦略
- [02-container-deployment.md](./02-container-deployment.md) — ECS/Kubernetes でのコンテナデプロイ
- [03-release-management.md](./03-release-management.md) — セマンティックバージョニングとリリース管理

---

## 参考文献

1. **AWS Well-Architected Framework** — https://docs.aws.amazon.com/wellarchitected/ — クラウドアーキテクチャのベストプラクティス
2. **Vercel Documentation** — https://vercel.com/docs — Vercel の公式ドキュメント
3. **Cloudflare Workers Documentation** — https://developers.cloudflare.com/workers/ — Workers の公式リファレンス
4. **AWS SAM Developer Guide** — https://docs.aws.amazon.com/serverless-application-model/ — SAM によるサーバーレスデプロイ

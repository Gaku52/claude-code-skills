# Next.js Deployment 完全ガイド

## 目次
1. [Vercel デプロイメント](#vercel-デプロイメント)
2. [セルフホスティング](#セルフホスティング)
3. [Docker コンテナ化](#docker-コンテナ化)
4. [AWS デプロイメント](#aws-デプロイメント)
5. [Cloudflare Pages](#cloudflare-pages)
6. [パフォーマンス最適化](#パフォーマンス最適化)
7. [環境変数管理](#環境変数管理)
8. [モニタリング](#モニタリング)

---

## Vercel デプロイメント

### Vercel CLI によるデプロイ

```bash
# Vercel CLIのインストール
npm i -g vercel

# プロジェクトのセットアップ
vercel

# Preview デプロイ
vercel

# Production デプロイ
vercel --prod
```

### GitHub 統合による自動デプロイ

```yaml
# .github/workflows/vercel-deploy.yml

name: Vercel Deploy

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install Vercel CLI
        run: npm install -g vercel

      - name: Pull Vercel Environment
        run: |
          if [ "${{ github.ref }}" == "refs/heads/main" ]; then
            vercel pull --yes --environment=production --token=${{ secrets.VERCEL_TOKEN }}
          else
            vercel pull --yes --environment=preview --token=${{ secrets.VERCEL_TOKEN }}
          fi

      - name: Build Project
        run: vercel build ${{ github.ref == 'refs/heads/main' && '--prod' || '' }} --token=${{ secrets.VERCEL_TOKEN }}

      - name: Deploy to Vercel
        id: deploy
        run: |
          if [ "${{ github.ref }}" == "refs/heads/main" ]; then
            echo "url=$(vercel deploy --prebuilt --prod --token=${{ secrets.VERCEL_TOKEN }})" >> $GITHUB_OUTPUT
          else
            echo "url=$(vercel deploy --prebuilt --token=${{ secrets.VERCEL_TOKEN }})" >> $GITHUB_OUTPUT
          fi

      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `✅ Preview deployed to: ${{ steps.deploy.outputs.url }}`
            })
```

### vercel.json 設定

```json
{
  "buildCommand": "npm run build",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "nextjs",
  "outputDirectory": ".next",
  "regions": ["iad1"],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/$1"
    }
  ],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "X-XSS-Protection",
          "value": "1; mode=block"
        }
      ]
    }
  ],
  "redirects": [
    {
      "source": "/old-blog/:slug",
      "destination": "/blog/:slug",
      "permanent": true
    }
  ],
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://api.example.com/:path*"
    }
  ],
  "crons": [
    {
      "path": "/api/cron/daily",
      "schedule": "0 0 * * *"
    }
  ]
}
```

### Environment Variables（Vercel）

```bash
# Vercel Dashboard → Settings → Environment Variables

# または CLI
vercel env add NEXT_PUBLIC_API_URL production
vercel env add DATABASE_URL production

# .env.local（ローカル開発用、Gitにコミットしない）
NEXT_PUBLIC_API_URL=http://localhost:3000
DATABASE_URL=postgresql://localhost:5432/myapp
```

---

## セルフホスティング

### Standalone Output

```javascript
// next.config.js

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  // 必要に応じて
  experimental: {
    outputFileTracingRoot: path.join(__dirname, '../../'),
  },
};

module.exports = nextConfig;
```

```bash
# ビルド
npm run build

# 生成されるファイル構造
.next/
├── standalone/
│   ├── .next/
│   ├── node_modules/
│   ├── package.json
│   └── server.js
├── static/
└── ...

# サーバー起動
cd .next/standalone
node server.js

# または環境変数を指定
PORT=3000 HOSTNAME=0.0.0.0 node server.js
```

### PM2 によるプロセス管理

```javascript
// ecosystem.config.js

module.exports = {
  apps: [
    {
      name: 'nextjs-app',
      script: '.next/standalone/server.js',
      instances: 'max',
      exec_mode: 'cluster',
      env: {
        NODE_ENV: 'production',
        PORT: 3000,
        HOSTNAME: '0.0.0.0',
      },
      env_production: {
        NODE_ENV: 'production',
      },
    },
  ],
};
```

```bash
# PM2のインストール
npm install -g pm2

# アプリケーション起動
pm2 start ecosystem.config.js --env production

# ステータス確認
pm2 status

# ログ確認
pm2 logs nextjs-app

# 再起動
pm2 restart nextjs-app

# 停止
pm2 stop nextjs-app

# スタートアップスクリプト生成（サーバー再起動時に自動起動）
pm2 startup
pm2 save
```

### Nginx リバースプロキシ

```nginx
# /etc/nginx/sites-available/nextjs-app

upstream nextjs_upstream {
  server 127.0.0.1:3000;
  server 127.0.0.1:3001;
  server 127.0.0.1:3002;
  keepalive 64;
}

server {
  listen 80;
  server_name example.com www.example.com;

  # HTTP to HTTPS redirect
  return 301 https://$server_name$request_uri;
}

server {
  listen 443 ssl http2;
  server_name example.com www.example.com;

  # SSL証明書（Let's Encrypt）
  ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

  # セキュリティヘッダー
  add_header X-Frame-Options "SAMEORIGIN" always;
  add_header X-Content-Type-Options "nosniff" always;
  add_header X-XSS-Protection "1; mode=block" always;
  add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

  # 静的ファイルのキャッシュ
  location /_next/static {
    proxy_pass http://nextjs_upstream;
    proxy_cache_valid 200 365d;
    add_header Cache-Control "public, max-age=31536000, immutable";
  }

  location /static {
    proxy_pass http://nextjs_upstream;
    proxy_cache_valid 200 365d;
    add_header Cache-Control "public, max-age=31536000, immutable";
  }

  # Next.js アプリケーション
  location / {
    proxy_pass http://nextjs_upstream;
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

---

## Docker コンテナ化

### Multi-stage Dockerfile

```dockerfile
# Dockerfile

FROM node:20-alpine AS base

# Dependencies
FROM base AS deps
RUN apk add --no-cache libc6-compat

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

# Builder
FROM base AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# 環境変数（ビルド時）
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

# Next.js テレメトリ無効化
ENV NEXT_TELEMETRY_DISABLED 1

RUN npm run build

# Runner
FROM base AS runner

WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# 必要なファイルのみコピー
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
```

### Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  nextjs:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        NEXT_PUBLIC_API_URL: https://api.example.com
    ports:
      - '3000:3000'
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://postgres:password@db:5432/myapp
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: myapp
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - '80:80'
      - '443:443'
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - nextjs
    restart: unless-stopped

volumes:
  postgres_data:
```

```bash
# ビルドと起動
docker-compose up -d

# ログ確認
docker-compose logs -f nextjs

# 停止
docker-compose down

# 完全削除（ボリューム含む）
docker-compose down -v
```

---

## AWS デプロイメント

### AWS Amplify

```yaml
# amplify.yml

version: 1
frontend:
  phases:
    preBuild:
      commands:
        - npm ci
    build:
      commands:
        - npm run build
  artifacts:
    baseDirectory: .next
    files:
      - '**/*'
  cache:
    paths:
      - node_modules/**/*
      - .next/cache/**/*
```

### AWS ECS + Fargate

```yaml
# .github/workflows/aws-ecs-deploy.yml

name: Deploy to AWS ECS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push Docker image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: nextjs-app
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build \
            --build-arg NEXT_PUBLIC_API_URL=${{ secrets.NEXT_PUBLIC_API_URL }} \
            -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
            -t $ECR_REGISTRY/$ECR_REPOSITORY:latest \
            .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

      - name: Update ECS task definition
        id: task-def
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: task-definition.json
          container-name: nextjs-app
          image: ${{ steps.login-ecr.outputs.registry }}/nextjs-app:${{ github.sha }}

      - name: Deploy to Amazon ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v1
        with:
          task-definition: ${{ steps.task-def.outputs.task-definition }}
          service: nextjs-app-service
          cluster: nextjs-app-cluster
          wait-for-service-stability: true
```

### AWS Lambda + API Gateway (Serverless)

```javascript
// next.config.js

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  images: {
    unoptimized: true, // Lambda環境ではImage Optimizationは無効化
  },
};

module.exports = nextConfig;
```

```bash
# Serverless Frameworkのインストール
npm install -g serverless

# プラグインのインストール
npm install --save-dev serverless-nextjs-plugin
```

```yaml
# serverless.yml

service: nextjs-app

provider:
  name: aws
  runtime: nodejs20.x
  region: us-east-1
  memorySize: 1024
  timeout: 30
  environment:
    NODE_ENV: production
    DATABASE_URL: ${env:DATABASE_URL}

plugins:
  - serverless-nextjs-plugin

custom:
  nextjs:
    # カスタム設定
    minifyHandlers: true
    enableHTTPCompression: true

functions:
  nextjs:
    handler: .next/serverless/pages/_app.render
    events:
      - http:
          path: /
          method: any
      - http:
          path: /{proxy+}
          method: any
```

---

## Cloudflare Pages

### Cloudflare Pages デプロイ

```yaml
# .github/workflows/cloudflare-pages.yml

name: Deploy to Cloudflare Pages

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build
        env:
          NEXT_PUBLIC_API_URL: ${{ secrets.NEXT_PUBLIC_API_URL }}

      - name: Publish to Cloudflare Pages
        uses: cloudflare/pages-action@v1
        with:
          apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          projectName: nextjs-app
          directory: .next
          gitHubToken: ${{ secrets.GITHUB_TOKEN }}
```

### Cloudflare Workers 統合

```javascript
// next.config.js

const nextConfig = {
  // Cloudflare Pages用の設定
  images: {
    loader: 'custom',
    loaderFile: './cloudflare-image-loader.js',
  },
};

module.exports = nextConfig;
```

```javascript
// cloudflare-image-loader.js

export default function cloudflareImageLoader({ src, width, quality }) {
  const params = [`width=${width}`];
  if (quality) {
    params.push(`quality=${quality}`);
  }
  return `/cdn-cgi/image/${params.join(',')}${src}`;
}
```

---

## パフォーマンス最適化

### next.config.js 最適化設定

```javascript
// next.config.js

/** @type {import('next').NextConfig} */
const nextConfig = {
  // 圧縮
  compress: true,

  // 画像最適化
  images: {
    formats: ['image/avif', 'image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    minimumCacheTTL: 60,
    dangerouslyAllowSVG: true,
    contentDispositionType: 'attachment',
    contentSecurityPolicy: "default-src 'self'; script-src 'none'; sandbox;",
  },

  // フォント最適化
  optimizeFonts: true,

  // SWC Minification
  swcMinify: true,

  // Strict Mode
  reactStrictMode: true,

  // Experimental features
  experimental: {
    optimizeCss: true,
    optimizePackageImports: ['@mui/material', 'lodash'],
  },

  // Webpack設定
  webpack: (config, { dev, isServer }) => {
    // Production ビルド最適化
    if (!dev && !isServer) {
      config.optimization.splitChunks = {
        chunks: 'all',
        cacheGroups: {
          default: false,
          vendors: false,
          commons: {
            name: 'commons',
            chunks: 'all',
            minChunks: 2,
          },
        },
      };
    }

    return config;
  },
};

module.exports = nextConfig;
```

### Bundle Analyzer

```bash
npm install --save-dev @next/bundle-analyzer
```

```javascript
// next.config.js

const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
});

module.exports = withBundleAnalyzer({
  // 既存の設定
});

// 実行
// ANALYZE=true npm run build
```

---

## 環境変数管理

### 環境変数の種類

```bash
# .env.local（ローカル開発、Gitにコミットしない）
DATABASE_URL=postgresql://localhost:5432/dev
API_SECRET=secret123
NEXT_PUBLIC_API_URL=http://localhost:3000

# .env.development（開発環境）
NEXT_PUBLIC_ANALYTICS_ID=dev-analytics-id

# .env.production（本番環境）
NEXT_PUBLIC_ANALYTICS_ID=prod-analytics-id

# .env（全環境共通のデフォルト値）
NEXT_PUBLIC_APP_NAME=My App
```

### 環境変数の検証

```typescript
// lib/env.ts

import { z } from 'zod';

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'test', 'production']),
  DATABASE_URL: z.string().url(),
  API_SECRET: z.string().min(1),
  NEXT_PUBLIC_API_URL: z.string().url(),
  NEXT_PUBLIC_ANALYTICS_ID: z.string().optional(),
});

export const env = envSchema.parse(process.env);

// 型安全な環境変数
// env.DATABASE_URL // string
// env.NEXT_PUBLIC_API_URL // string
```

---

## モニタリング

### Sentry 統合

```bash
npm install @sentry/nextjs
```

```javascript
// sentry.client.config.ts

import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  tracesSampleRate: 1.0,
  environment: process.env.NODE_ENV,
});
```

```javascript
// sentry.server.config.ts

import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  tracesSampleRate: 1.0,
  environment: process.env.NODE_ENV,
});
```

### Vercel Analytics

```bash
npm install @vercel/analytics
```

```typescript
// app/layout.tsx

import { Analytics } from '@vercel/analytics/react';

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
```

### Web Vitals モニタリング

```typescript
// app/web-vitals.tsx

'use client';

import { useReportWebVitals } from 'next/web-vitals';

export function WebVitals() {
  useReportWebVitals((metric) => {
    // Google Analytics
    window.gtag('event', metric.name, {
      value: Math.round(metric.name === 'CLS' ? metric.value * 1000 : metric.value),
      event_label: metric.id,
      non_interaction: true,
    });

    // Custom endpoint
    fetch('/api/analytics', {
      method: 'POST',
      body: JSON.stringify(metric),
    });
  });

  return null;
}
```

---

このガイドでは、Next.jsアプリケーションのデプロイメント方法について、Vercelからセルフホスティング、Docker、AWS、Cloudflare Pagesまで、包括的に解説しました。環境変数管理、パフォーマンス最適化、モニタリングまで含めることで、本番環境での安定稼働を実現できます。

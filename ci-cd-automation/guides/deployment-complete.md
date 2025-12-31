# デプロイメント自動化 完全ガイド

## 目次
1. [デプロイメント戦略](#デプロイメント戦略)
2. [環境管理](#環境管理)
3. [iOS/Android デプロイメント](#iosandroid-デプロイメント)
4. [Web デプロイメント](#web-デプロイメント)
5. [Backend デプロイメント](#backend-デプロイメント)
6. [ロールバック戦略](#ロールバック戦略)
7. [モニタリング](#モニタリング)
8. [セキュリティ](#セキュリティ)

---

## デプロイメント戦略

### デプロイメントパターン

#### 1. Blue-Green デプロイメント

```yaml
# .github/workflows/blue-green-deploy.yml

name: Blue-Green Deployment

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build
        run: |
          docker build -t myapp:${{ github.sha }} .

      - name: Deploy to Green Environment
        run: |
          # Green環境にデプロイ
          kubectl set image deployment/myapp-green \
            myapp=myapp:${{ github.sha }} \
            -n production

      - name: Health Check
        run: |
          # Green環境のヘルスチェック
          ./scripts/health-check.sh green

      - name: Switch Traffic to Green
        if: success()
        run: |
          # トラフィックをGreenに切り替え
          kubectl patch service myapp \
            -p '{"spec":{"selector":{"version":"green"}}}' \
            -n production

      - name: Rollback on Failure
        if: failure()
        run: |
          # 失敗時はBlueに戻す
          kubectl patch service myapp \
            -p '{"spec":{"selector":{"version":"blue"}}}' \
            -n production
```

#### 2. Canary デプロイメント

```yaml
# .github/workflows/canary-deploy.yml

name: Canary Deployment

on:
  push:
    branches: [main]

jobs:
  canary-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Deploy Canary (10%)
        run: |
          # Canary版を10%のトラフィックで展開
          kubectl apply -f k8s/canary-10percent.yaml

      - name: Monitor Canary
        run: |
          # 10分間モニタリング
          ./scripts/monitor-canary.sh 600

      - name: Increase to 50%
        if: success()
        run: |
          kubectl apply -f k8s/canary-50percent.yaml
          ./scripts/monitor-canary.sh 600

      - name: Full Rollout
        if: success()
        run: |
          kubectl apply -f k8s/production.yaml

      - name: Rollback Canary
        if: failure()
        run: |
          kubectl delete -f k8s/canary-10percent.yaml
```

#### 3. Rolling デプロイメント

```yaml
# k8s/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2          # 同時に追加できるPod数
      maxUnavailable: 1    # 同時に停止できるPod数
  template:
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 20
```

### フィーチャーフラグによる段階的リリース

```typescript
// src/feature-flags/index.ts

import { LaunchDarkly } from '@launchdarkly/node-server-sdk';

class FeatureFlagService {
  private client: LaunchDarkly.LDClient;

  async initialize() {
    this.client = LaunchDarkly.init(process.env.LAUNCHDARKLY_SDK_KEY!);
    await this.client.waitForInitialization();
  }

  async isFeatureEnabled(
    featureKey: string,
    user: { key: string; email?: string; custom?: any }
  ): Promise<boolean> {
    return await this.client.variation(featureKey, user, false);
  }

  async getFeatureVariant(
    featureKey: string,
    user: { key: string }
  ): Promise<string> {
    return await this.client.variation(featureKey, user, 'control');
  }
}

// 使用例
const flags = new FeatureFlagService();
await flags.initialize();

// ユーザーごとに機能を有効化
app.get('/api/data', async (req, res) => {
  const user = { key: req.user.id, email: req.user.email };

  if (await flags.isFeatureEnabled('new-dashboard', user)) {
    // 新しいダッシュボードを表示
    return res.json(await getNewDashboard());
  } else {
    // 既存のダッシュボードを表示
    return res.json(await getOldDashboard());
  }
});
```

---

## 環境管理

### 環境別設定

```typescript
// config/environments.ts

export const environments = {
  development: {
    apiBaseUrl: 'http://localhost:3000',
    databaseUrl: 'postgresql://localhost:5432/myapp_dev',
    logLevel: 'debug',
    enableCORS: true,
  },
  staging: {
    apiBaseUrl: 'https://staging-api.example.com',
    databaseUrl: process.env.DATABASE_URL,
    logLevel: 'info',
    enableCORS: true,
  },
  production: {
    apiBaseUrl: 'https://api.example.com',
    databaseUrl: process.env.DATABASE_URL,
    logLevel: 'warn',
    enableCORS: false,
  },
};

export function getConfig() {
  const env = process.env.NODE_ENV || 'development';
  return environments[env];
}
```

### Secretsの管理

```yaml
# GitHub Secrets の設定

# Settings → Secrets and variables → Actions → New repository secret

# Development
DEV_DATABASE_URL
DEV_API_KEY
DEV_AWS_ACCESS_KEY_ID
DEV_AWS_SECRET_ACCESS_KEY

# Staging
STAGING_DATABASE_URL
STAGING_API_KEY
STAGING_AWS_ACCESS_KEY_ID
STAGING_AWS_SECRET_ACCESS_KEY

# Production
PROD_DATABASE_URL
PROD_API_KEY
PROD_AWS_ACCESS_KEY_ID
PROD_AWS_SECRET_ACCESS_KEY
```

```yaml
# .github/workflows/deploy.yml

jobs:
  deploy-staging:
    environment: staging
    steps:
      - name: Deploy to Staging
        env:
          DATABASE_URL: ${{ secrets.STAGING_DATABASE_URL }}
          API_KEY: ${{ secrets.STAGING_API_KEY }}
        run: |
          ./deploy.sh staging

  deploy-production:
    environment: production
    needs: deploy-staging
    steps:
      - name: Deploy to Production
        env:
          DATABASE_URL: ${{ secrets.PROD_DATABASE_URL }}
          API_KEY: ${{ secrets.PROD_API_KEY }}
        run: |
          ./deploy.sh production
```

### AWS Secrets Manager統合

```typescript
// src/config/secrets.ts

import { SecretsManagerClient, GetSecretValueCommand } from '@aws-sdk/client-secrets-manager';

const client = new SecretsManagerClient({ region: 'us-east-1' });

export async function getSecret(secretName: string): Promise<any> {
  try {
    const response = await client.send(
      new GetSecretValueCommand({ SecretId: secretName })
    );

    if (response.SecretString) {
      return JSON.parse(response.SecretString);
    }
  } catch (error) {
    console.error('Error retrieving secret:', error);
    throw error;
  }
}

// 使用例
const dbCredentials = await getSecret('production/database');
const apiKeys = await getSecret('production/api-keys');
```

---

## iOS/Android デプロイメント

### iOS - App Store Connect

```ruby
# fastlane/Fastfile

platform :ios do
  desc "Deploy to App Store"
  lane :deploy do
    # 1. 環境変数の確認
    ensure_env_vars(
      env_vars: ['MATCH_PASSWORD', 'FASTLANE_APPLE_APPLICATION_SPECIFIC_PASSWORD']
    )

    # 2. Git状態の確認
    ensure_git_status_clean
    ensure_git_branch(branch: 'main')

    # 3. テスト実行
    run_tests(
      scheme: "MyApp",
      devices: ["iPhone 15 Pro"],
      code_coverage: true
    )

    # 4. バージョン管理
    increment_build_number(
      build_number: latest_testflight_build_number + 1
    )

    # 5. 証明書・プロファイル同期
    match(type: "appstore", readonly: true)

    # 6. ビルド
    build_app(
      scheme: "MyApp",
      configuration: "Release",
      export_method: "app-store",
      export_options: {
        provisioningProfiles: {
          "com.company.myapp" => "match AppStore com.company.myapp"
        }
      }
    )

    # 7. dSYMアップロード（Crashlytics）
    upload_symbols_to_crashlytics(
      gsp_path: "./MyApp/GoogleService-Info.plist"
    )

    # 8. TestFlightアップロード
    upload_to_testflight(
      skip_submission: false,
      skip_waiting_for_build_processing: true,
      distribute_external: true,
      groups: ["Internal Testers"],
      changelog: changelog_from_git_commits(
        between: [ENV['GIT_PREVIOUS_SUCCESSFUL_COMMIT'] || 'HEAD^^^^^', 'HEAD'],
        pretty: '- %s'
      )
    )

    # 9. App Store申請
    upload_to_app_store(
      submit_for_review: true,
      automatic_release: false,
      force: true,
      skip_metadata: false,
      skip_screenshots: false,
      phased_release: true,
      submission_information: {
        add_id_info_uses_idfa: true,
        export_compliance_uses_encryption: false
      }
    )

    # 10. Gitタグ作成
    version = get_version_number
    build = get_build_number
    add_git_tag(tag: "ios/v#{version}-#{build}")
    push_git_tags

    # 11. Slack通知
    slack(
      message: "iOS v#{version} (#{build}) submitted to App Store! 🚀",
      success: true,
      channel: "#releases",
      payload: {
        "Build Time" => Time.now.to_s,
        "Built by" => ENV['USER']
      }
    )
  end
end
```

### Android - Google Play

```groovy
// build.gradle

android {
    defaultConfig {
        versionCode getVersionCodeFromGit()
        versionName "1.0.0"
    }

    signingConfigs {
        release {
            storeFile file(System.getenv("KEYSTORE_FILE") ?: "keystore.jks")
            storePassword System.getenv("KEYSTORE_PASSWORD")
            keyAlias System.getenv("KEY_ALIAS")
            keyPassword System.getenv("KEY_PASSWORD")
        }
    }

    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
}

def getVersionCodeFromGit() {
    def cmd = 'git rev-list --count HEAD'
    return cmd.execute().text.trim().toInteger()
}
```

```ruby
# fastlane/Fastfile (Android)

platform :android do
  desc "Deploy to Google Play"
  lane :deploy do
    # 1. テスト実行
    gradle(task: "test")

    # 2. ビルド
    gradle(
      task: "bundle",
      build_type: "Release"
    )

    # 3. Internal Testingトラックにアップロード
    upload_to_play_store(
      track: 'internal',
      aab: 'app/build/outputs/bundle/release/app-release.aab',
      skip_upload_metadata: true,
      skip_upload_images: true,
      skip_upload_screenshots: true
    )

    # 4. Beta（Open Testing）にプロモート
    upload_to_play_store(
      track: 'internal',
      track_promote_to: 'beta',
      skip_upload_aab: true
    )

    # 5. Production リリース（段階的展開）
    upload_to_play_store(
      track: 'beta',
      track_promote_to: 'production',
      rollout: '0.1',  # 10%から開始
      skip_upload_aab: true
    )
  end
end
```

---

## Web デプロイメント

### Vercel デプロイメント

```yaml
# .github/workflows/vercel-deploy.yml

name: Deploy to Vercel

on:
  push:
    branches: [main, develop]

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

      - name: Deploy to Preview (develop)
        if: github.ref == 'refs/heads/develop'
        run: |
          vercel pull --yes --environment=preview --token=${{ secrets.VERCEL_TOKEN }}
          vercel build --token=${{ secrets.VERCEL_TOKEN }}
          vercel deploy --prebuilt --token=${{ secrets.VERCEL_TOKEN }}

      - name: Deploy to Production (main)
        if: github.ref == 'refs/heads/main'
        run: |
          vercel pull --yes --environment=production --token=${{ secrets.VERCEL_TOKEN }}
          vercel build --prod --token=${{ secrets.VERCEL_TOKEN }}
          vercel deploy --prebuilt --prod --token=${{ secrets.VERCEL_TOKEN }}
```

### AWS Amplify デプロイメント

```yaml
# amplify.yml

version: 1
applications:
  - frontend:
      phases:
        preBuild:
          commands:
            - npm ci
        build:
          commands:
            - npm run build
      artifacts:
        baseDirectory: dist
        files:
          - '**/*'
      cache:
        paths:
          - node_modules/**/*
    appRoot: frontend

  - backend:
      phases:
        build:
          commands:
            - npm ci
            - npm run build
      artifacts:
        baseDirectory: backend/dist
        files:
          - '**/*'
    appRoot: backend
```

### Cloudflare Pages デプロイメント

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

      - name: Build
        run: |
          npm ci
          npm run build

      - name: Publish to Cloudflare Pages
        uses: cloudflare/pages-action@v1
        with:
          apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          projectName: myapp
          directory: dist
          gitHubToken: ${{ secrets.GITHUB_TOKEN }}
```

---

## Backend デプロイメント

### Docker + AWS ECS

```dockerfile
# Dockerfile (Multi-stage build)

# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage
FROM node:20-alpine

WORKDIR /app

COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package*.json ./

EXPOSE 3000

CMD ["node", "dist/main.js"]
```

```yaml
# .github/workflows/ecs-deploy.yml

name: Deploy to ECS

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
          ECR_REPOSITORY: myapp
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

      - name: Update ECS service
        run: |
          aws ecs update-service \
            --cluster myapp-cluster \
            --service myapp-service \
            --force-new-deployment

      - name: Wait for deployment
        run: |
          aws ecs wait services-stable \
            --cluster myapp-cluster \
            --services myapp-service
```

### Kubernetes デプロイメント

```yaml
# k8s/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
        version: v1
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: myapp-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 15
          periodSeconds: 20

---
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
  namespace: production
spec:
  selector:
    app: myapp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3000
  type: LoadBalancer
```

```yaml
# .github/workflows/k8s-deploy.yml

name: Deploy to Kubernetes

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up kubectl
        uses: azure/setup-kubectl@v3

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG }}" > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to Kubernetes
        run: |
          kubectl apply -f k8s/
          kubectl rollout status deployment/myapp -n production
```

---

## ロールバック戦略

### Git Revert

```bash
# 最新のコミットをrevert
git revert HEAD
git push origin main

# 特定のコミットをrevert
git revert <commit-hash>
git push origin main
```

### Kubernetes ロールバック

```bash
# デプロイメント履歴を確認
kubectl rollout history deployment/myapp -n production

# 前のバージョンにロールバック
kubectl rollout undo deployment/myapp -n production

# 特定のリビジョンにロールバック
kubectl rollout undo deployment/myapp --to-revision=2 -n production

# ロールバック状況を監視
kubectl rollout status deployment/myapp -n production
```

### AWS ECS ロールバック

```bash
# 現在のタスク定義を確認
aws ecs describe-services \
  --cluster myapp-cluster \
  --services myapp-service

# 前のタスク定義にロールバック
aws ecs update-service \
  --cluster myapp-cluster \
  --service myapp-service \
  --task-definition myapp:123  # 前のリビジョン番号
```

### 自動ロールバック

```yaml
# .github/workflows/auto-rollback.yml

name: Deploy with Auto-Rollback

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        id: deploy
        run: |
          kubectl apply -f k8s/deployment.yaml
          kubectl rollout status deployment/myapp -n production

      - name: Health Check
        id: health-check
        run: |
          sleep 30
          ./scripts/health-check.sh

      - name: Rollback on Failure
        if: failure()
        run: |
          echo "Health check failed, rolling back..."
          kubectl rollout undo deployment/myapp -n production
          kubectl rollout status deployment/myapp -n production

      - name: Notify on Rollback
        if: failure()
        run: |
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -d '{"text":"🚨 Deployment failed and rolled back!"}'
```

---

## モニタリング

### デプロイメントメトリクス

```typescript
// src/monitoring/deployment-metrics.ts

import { CloudWatch } from '@aws-sdk/client-cloudwatch';

const cloudwatch = new CloudWatch({ region: 'us-east-1' });

export async function recordDeployment(
  environment: string,
  version: string,
  status: 'success' | 'failure'
) {
  await cloudwatch.putMetricData({
    Namespace: 'MyApp/Deployments',
    MetricData: [
      {
        MetricName: 'DeploymentCount',
        Value: 1,
        Unit: 'Count',
        Dimensions: [
          { Name: 'Environment', Value: environment },
          { Name: 'Status', Value: status },
        ],
      },
    ],
  });
}

// デプロイメント時間の記録
export async function recordDeploymentDuration(
  environment: string,
  duration: number
) {
  await cloudwatch.putMetricData({
    Namespace: 'MyApp/Deployments',
    MetricData: [
      {
        MetricName: 'DeploymentDuration',
        Value: duration,
        Unit: 'Seconds',
        Dimensions: [{ Name: 'Environment', Value: environment }],
      },
    ],
  });
}
```

### ヘルスチェック

```typescript
// src/health/health-check.ts

import express from 'express';

const router = express.Router();

router.get('/health', async (req, res) => {
  const health = {
    uptime: process.uptime(),
    message: 'OK',
    timestamp: Date.now(),
    checks: {
      database: await checkDatabase(),
      redis: await checkRedis(),
      externalAPI: await checkExternalAPI(),
    },
  };

  const isHealthy = Object.values(health.checks).every((check) => check.status === 'up');

  res.status(isHealthy ? 200 : 503).json(health);
});

async function checkDatabase(): Promise<{ status: string }> {
  try {
    await db.raw('SELECT 1');
    return { status: 'up' };
  } catch (error) {
    return { status: 'down' };
  }
}

async function checkRedis(): Promise<{ status: string }> {
  try {
    await redis.ping();
    return { status: 'up' };
  } catch (error) {
    return { status: 'down' };
  }
}
```

---

## セキュリティ

### デプロイメント承認

```yaml
# .github/workflows/production-deploy.yml

name: Production Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://app.example.com
    steps:
      - name: Deploy
        run: ./deploy.sh production

# Settings → Environments → production → Required reviewers を設定
# デプロイ前に承認が必要
```

### 監査ログ

```typescript
// src/audit/deployment-audit.ts

import { DynamoDB } from '@aws-sdk/client-dynamodb';

const dynamodb = new DynamoDB({ region: 'us-east-1' });

export async function logDeployment(deployment: {
  environment: string;
  version: string;
  deployedBy: string;
  timestamp: number;
  gitCommit: string;
}) {
  await dynamodb.putItem({
    TableName: 'DeploymentAuditLog',
    Item: {
      id: { S: `${deployment.environment}-${deployment.timestamp}` },
      environment: { S: deployment.environment },
      version: { S: deployment.version },
      deployedBy: { S: deployment.deployedBy },
      timestamp: { N: deployment.timestamp.toString() },
      gitCommit: { S: deployment.gitCommit },
    },
  });
}
```

---

このガイドでは、モダンなデプロイメント戦略から、iOS/Android/Web/Backendの具体的なデプロイメント手法、ロールバック戦略、モニタリング、セキュリティまで、包括的なデプロイメント自動化のベストプラクティスを解説しました。

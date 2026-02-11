# コンテナデプロイ

> ECS、Kubernetes、ArgoCD を活用したコンテナベースのデプロイパイプラインを構築し、スケーラブルで再現性の高いデプロイを実現する

## この章で学ぶこと

1. **Docker イメージの最適化とレジストリ管理** — マルチステージビルド、イメージサイズ削減、ECR/GHCR の運用
2. **ECS (Fargate) によるコンテナデプロイ** — タスク定義、サービス設定、CI/CD パイプライン構築
3. **Kubernetes + ArgoCD による GitOps デプロイ** — マニフェスト管理、自動同期、Progressive Delivery

---

## 1. コンテナデプロイの全体像

```
┌─────────────────────────────────────────────────────────┐
│            コンテナデプロイ パイプライン                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ソースコード    ビルド         レジストリ    オーケストレータ │
│  ┌──────┐    ┌──────────┐   ┌────────┐   ┌──────────┐ │
│  │ Git  │───►│ Docker   │──►│ ECR /  │──►│ ECS /    │ │
│  │ Repo │    │ Build    │   │ GHCR   │   │ K8s      │ │
│  └──────┘    └──────────┘   └────────┘   └──────────┘ │
│       │                                       │        │
│       │    GitOps の場合                       │        │
│       │    ┌──────────┐                       │        │
│       └───►│ ArgoCD   │──── 自動同期 ────────►│        │
│            └──────────┘                                │
│                                                         │
│  [CI] GitHub Actions          [CD] ArgoCD / ECS Deploy  │
│  - テスト実行                  - ローリングアップデート     │
│  - イメージビルド              - ヘルスチェック             │
│  - 脆弱性スキャン              - 自動ロールバック          │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Docker イメージの最適化

```dockerfile
# Dockerfile — マルチステージビルド (Node.js)
# ============================================
# Stage 1: 依存関係のインストール
# ============================================
FROM node:20-alpine AS deps
WORKDIR /app

# パッケージファイルだけを先にコピー (キャッシュ活用)
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# ============================================
# Stage 2: ビルド
# ============================================
FROM node:20-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ============================================
# Stage 3: 本番イメージ (最小構成)
# ============================================
FROM node:20-alpine AS runner
WORKDIR /app

# セキュリティ: 非root ユーザーで実行
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 appuser

# 本番依存関係とビルド成果物のみコピー
COPY --from=deps /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./

USER appuser

EXPOSE 3000
ENV NODE_ENV=production

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

CMD ["node", "dist/server.js"]
```

```yaml
# .dockerignore
node_modules
.git
.github
.env*
*.md
Dockerfile
docker-compose*.yml
coverage
.nyc_output
dist
```

---

## 3. GitHub Actions — イメージビルドとプッシュ

```yaml
# .github/workflows/build-and-push.yml
name: Build and Push Container Image

on:
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    outputs:
      image-tag: ${{ steps.meta.outputs.version }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=ref,event=tag
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and Push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
```

---

## 4. ECS (Fargate) デプロイ

```json
// ecs-task-definition.json
{
  "family": "myapp",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "app",
      "image": "ghcr.io/myorg/myapp:latest",
      "portMappings": [
        {
          "containerPort": 3000,
          "protocol": "tcp"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "wget --spider http://localhost:3000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/myapp",
          "awslogs-region": "ap-northeast-1",
          "awslogs-stream-prefix": "app"
        }
      },
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:ap-northeast-1:123456789012:secret:myapp/db-url"
        }
      ],
      "environment": [
        { "name": "NODE_ENV", "value": "production" },
        { "name": "PORT", "value": "3000" }
      ]
    }
  ]
}
```

```
ECS Fargate デプロイの流れ:

  GitHub Actions               ECR              ECS Service
      │                         │                    │
      │── docker build ───►     │                    │
      │── docker push ────►     │                    │
      │                         │                    │
      │── aws ecs               │                    │
      │   update-service ─────────────────────►      │
      │                         │                    │
      │                         │   ┌── 新タスク起動  │
      │                         │   │   (v2 イメージ) │
      │                         │   │                │
      │                         │   │   ヘルスチェック │
      │                         │   │   ┌─ OK ──►   │
      │                         │   │   │   旧タスク  │
      │                         │   │   │   停止     │
      │                         │   │   │            │
      │                         │   │   └─ NG ──►   │
      │                         │   │       ロール   │
      │                         │   │       バック   │
```

---

## 5. Kubernetes + ArgoCD (GitOps)

```yaml
# k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  labels:
    app: myapp
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      serviceAccountName: myapp
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
      containers:
        - name: app
          image: ghcr.io/myorg/myapp:abc1234
          ports:
            - containerPort: 3000
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 512Mi
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
          env:
            - name: NODE_ENV
              value: production
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: myapp-secrets
                  key: database-url
---
# k8s/base/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp
  ports:
    - port: 80
      targetPort: 3000
  type: ClusterIP
---
# k8s/base/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: myapp
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

```yaml
# argocd/application.yaml — ArgoCD Application 定義
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/myapp-k8s-manifests.git
    targetRevision: main
    path: k8s/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true        # 不要なリソースを自動削除
      selfHeal: true      # 手動変更を自動修正
    syncOptions:
      - CreateNamespace=true
    retry:
      limit: 3
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

---

## 6. 比較表

| 特性 | ECS (Fargate) | Kubernetes (EKS) | Docker Compose |
|------|--------------|-------------------|----------------|
| 管理負荷 | 低い | 高い | 最低 |
| スケーラビリティ | 高い | 最高 | 低い |
| 学習コスト | 中 | 高い | 低い |
| エコシステム | AWS 内完結 | 巨大 (CNCF) | 限定的 |
| コスト | 中 | 高い (コントロールプレーン有料) | 低い |
| GitOps 対応 | CodePipeline | ArgoCD / Flux | 困難 |
| 適用規模 | 中〜大規模 | 大規模 | 開発/小規模 |

| GitOps ツール比較 | ArgoCD | Flux | Jenkins X |
|-------------------|--------|------|-----------|
| UI ダッシュボード | 充実 | 基本的 | あり |
| マルチクラスタ | 対応 | 対応 | 限定的 |
| Helm 対応 | 対応 | 対応 | 対応 |
| Kustomize 対応 | 対応 | 対応 | 限定的 |
| RBAC | 細かい制御 | K8s RBAC | 独自 |
| コミュニティ | 大きい | 大きい | 小さい |

---

## 7. アンチパターン

### アンチパターン 1: latest タグの本番利用

```dockerfile
# 悪い例: latest タグで本番デプロイ
# どのバージョンが動いているか不明、ロールバック不可能
image: myapp:latest

# 良い例: Git SHA または セマンティックバージョンを使用
image: ghcr.io/myorg/myapp:a1b2c3d
image: ghcr.io/myorg/myapp:v1.2.3

# CI/CD でイメージタグを自動設定する仕組みを整備する
```

### アンチパターン 2: リソース制限なしでのデプロイ

```yaml
# 悪い例: リソース制限なし
containers:
  - name: app
    image: myapp:v1.0.0
    # resources 未設定 → メモリリークで Node 全体に影響

# 良い例: requests と limits を適切に設定
containers:
  - name: app
    image: myapp:v1.0.0
    resources:
      requests:        # スケジューリングの基準
        cpu: 100m      # 0.1 CPU コア
        memory: 128Mi
      limits:          # 超過時に制限/OOMKill
        cpu: 500m
        memory: 512Mi
```

---

## 8. FAQ

### Q1: ECS と Kubernetes、どちらを選ぶべきですか？

チームの Kubernetes 経験と運用負荷の許容度で判断します。AWS に閉じたシンプルなコンテナ運用なら ECS Fargate が管理負荷が低く始めやすいです。マルチクラウド、高度なトラフィック制御（Istio 等）、豊富な OSS エコシステムが必要なら Kubernetes を選択してください。EKS のコントロールプレーン費用（約$73/月）も考慮に入れましょう。

### Q2: ArgoCD の「自動同期」は常に有効にすべきですか？

開発環境では有効にして問題ありません。本番環境では `automated.prune: true` と `selfHeal: true` を慎重に検討してください。特に `prune` は Git リポジトリから削除されたリソースを自動削除するため、誤操作のリスクがあります。段階的に導入し、まずは手動同期（Sync ボタン）から始めて信頼性を確認することを推奨します。

### Q3: コンテナイメージの脆弱性スキャンはどのタイミングで行うべきですか？

**ビルド時**（CI パイプライン内）と**定期スキャン**の2段階が推奨です。ビルド時には `trivy image` や `docker scout` を実行し、Critical/High の脆弱性があればビルドを失敗させます。レジストリに保存済みのイメージも、新しい CVE が公開される可能性があるため、日次の定期スキャンを設定してください。

---

## まとめ

| 項目 | 要点 |
|------|------|
| マルチステージビルド | ビルド成果物のみを最終イメージに含め、サイズと攻撃面を最小化 |
| イメージタグ | 本番では Git SHA またはセマンティックバージョンを使用。latest 禁止 |
| ECS Fargate | AWS マネージド。タスク定義 + サービスで運用。管理負荷が低い |
| Kubernetes | 高い柔軟性。HPA でオートスケール、RBAC で権限管理 |
| ArgoCD (GitOps) | Git リポジトリを信頼の源泉とし、宣言的にデプロイ状態を管理 |
| リソース制限 | requests/limits を必ず設定。OOMKill やノード影響を防止 |

---

## 次に読むべきガイド

- [00-deployment-strategies.md](./00-deployment-strategies.md) — Blue-Green、Canary などのデプロイ戦略
- [01-cloud-deployment.md](./01-cloud-deployment.md) — AWS/Vercel/Cloudflare Workers
- [03-release-management.md](./03-release-management.md) — セマンティックバージョニングとリリース管理

---

## 参考文献

1. **Docker Documentation - Multi-stage builds** — https://docs.docker.com/build/building/multi-stage/ — マルチステージビルドの公式ガイド
2. **Amazon ECS Developer Guide** — https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ — ECS の公式ドキュメント
3. **ArgoCD Documentation** — https://argo-cd.readthedocs.io/ — GitOps ベースの CD ツール公式ドキュメント
4. **Kubernetes Best Practices** — Brendan Burns, Eddie Villalba, Dave Strebel, Lachlan Evenson (O'Reilly, 2019)

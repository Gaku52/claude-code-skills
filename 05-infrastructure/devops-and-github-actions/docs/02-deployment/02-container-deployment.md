# コンテナデプロイ

> ECS、Kubernetes、ArgoCD を活用したコンテナベースのデプロイパイプラインを構築し、スケーラブルで再現性の高いデプロイを実現する

## この章で学ぶこと

1. **Docker イメージの最適化とレジストリ管理** — マルチステージビルド、イメージサイズ削減、ECR/GHCR の運用
2. **ECS (Fargate) によるコンテナデプロイ** — タスク定義、サービス設定、CI/CD パイプライン構築
3. **Kubernetes + ArgoCD による GitOps デプロイ** — マニフェスト管理、自動同期、Progressive Delivery
4. **コンテナセキュリティとイメージ脆弱性管理** — スキャン自動化、ポリシー適用、セキュリティベストプラクティス
5. **マルチ環境対応とプロモーション戦略** — dev/staging/production 間のイメージプロモーション設計

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

### コンテナデプロイの基本原則

コンテナベースのデプロイには、従来の VM ベースデプロイと比較して以下の原則が重要になる。

1. **イミュータブルインフラストラクチャ**: コンテナイメージは一度ビルドしたら変更しない。環境差分は環境変数やシークレットで注入する
2. **宣言的構成管理**: デプロイの望ましい状態をコードで宣言し、オーケストレータがその状態を維持する
3. **再現性の保証**: 同じイメージタグを使えば、いつでもどこでも同じ環境を再現できる
4. **段階的ロールアウト**: 新バージョンを段階的にデプロイし、問題があれば即座にロールバックできる

```
コンテナデプロイのレイヤー構成:

  ┌─────────────────────────────────────────────┐
  │         アプリケーション層                      │
  │  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
  │  │ Service │  │ Service │  │ Service │    │
  │  │    A    │  │    B    │  │    C    │    │
  │  └────┬────┘  └────┬────┘  └────┬────┘    │
  ├───────┼────────────┼────────────┼──────────┤
  │       │ コンテナオーケストレーション層         │
  │  ┌────┴────────────┴────────────┴────┐     │
  │  │   ECS / Kubernetes / Nomad        │     │
  │  │   - スケジューリング               │     │
  │  │   - ヘルスチェック                 │     │
  │  │   - オートスケーリング              │     │
  │  │   - サービスディスカバリ            │     │
  │  └──────────────────────────────────┘     │
  ├────────────────────────────────────────────┤
  │         インフラストラクチャ層                 │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
  │  │ Compute  │  │ Network  │  │ Storage  │ │
  │  │ (EC2/    │  │ (VPC/    │  │ (EBS/    │ │
  │  │  Fargate)│  │  ALB)    │  │  EFS)    │ │
  │  └──────────┘  └──────────┘  └──────────┘ │
  └─────────────────────────────────────────────┘
```

---

## 2. Docker イメージの最適化

### 2.1 マルチステージビルドの基本

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

### 2.2 Go アプリケーションのマルチステージビルド

```dockerfile
# Dockerfile — Go マルチステージビルド (スクラッチイメージ)
# ============================================
# Stage 1: ビルド
# ============================================
FROM golang:1.22-alpine AS builder
WORKDIR /app

# 依存関係の先行ダウンロード (キャッシュ活用)
COPY go.mod go.sum ./
RUN go mod download

# ソースコードのコピーとビルド
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s -X main.version=$(git describe --tags 2>/dev/null || echo 'dev')" \
    -o /app/server ./cmd/server

# ============================================
# Stage 2: 最小イメージ (scratch)
# ============================================
FROM scratch

# CA 証明書 (HTTPS 通信に必要)
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# タイムゾーン情報
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# passwd ファイル (非root ユーザー)
COPY --from=builder /etc/passwd /etc/passwd

# バイナリのみコピー
COPY --from=builder /app/server /server

USER 1001

EXPOSE 8080

ENTRYPOINT ["/server"]
```

### 2.3 Python アプリケーションのマルチステージビルド

```dockerfile
# Dockerfile — Python (FastAPI) マルチステージビルド
# ============================================
# Stage 1: 依存関係のビルド
# ============================================
FROM python:3.12-slim AS builder
WORKDIR /app

# システム依存パッケージのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# 仮想環境の作成と依存関係のインストール
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: 本番イメージ
# ============================================
FROM python:3.12-slim AS runner
WORKDIR /app

# ランタイムに必要なライブラリのみインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# 非root ユーザー
RUN groupadd --system --gid 1001 appgroup && \
    useradd --system --uid 1001 --gid appgroup appuser

# 仮想環境をコピー
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# アプリケーションコードをコピー
COPY . .

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.4 .dockerignore の設定

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
.vscode
.idea
__pycache__
*.pyc
.pytest_cache
.mypy_cache
.tox
venv
.venv
```

### 2.5 イメージサイズ最適化のベストプラクティス

```
イメージサイズ比較:

  ベースイメージの選択による差:
  ┌────────────────────┬────────────┐
  │ ベースイメージ       │ サイズ     │
  ├────────────────────┼────────────┤
  │ node:20            │ ~1.1 GB    │
  │ node:20-slim       │ ~240 MB    │
  │ node:20-alpine     │ ~140 MB    │
  │ distroless/nodejs  │ ~130 MB    │
  ├────────────────────┼────────────┤
  │ python:3.12        │ ~1.0 GB    │
  │ python:3.12-slim   │ ~150 MB    │
  │ python:3.12-alpine │ ~60 MB     │
  ├────────────────────┼────────────┤
  │ golang:1.22        │ ~800 MB    │
  │ golang:1.22-alpine │ ~260 MB    │
  │ scratch (Go)       │ ~10-20 MB  │
  └────────────────────┴────────────┘

  最適化テクニック:
  1. Alpine / slim ベースイメージを使用
  2. マルチステージビルドで不要なツールを排除
  3. .dockerignore でビルドコンテキストを最小化
  4. レイヤーの統合 (RUN 命令を && で連結)
  5. apt-get install 後に rm -rf /var/lib/apt/lists/*
  6. --no-install-recommends で推奨パッケージを除外
  7. COPY の順序を工夫してキャッシュ効率を最大化
```

---

## 3. コンテナレジストリの管理

### 3.1 ECR (Elastic Container Registry) の構成

```yaml
# ecr-lifecycle-policy.json — イメージのライフサイクル管理
{
  "rules": [
    {
      "rulePriority": 1,
      "description": "セマンティックバージョンタグは保持",
      "selection": {
        "tagStatus": "tagged",
        "tagPrefixList": ["v"],
        "countType": "imageCountMoreThan",
        "countNumber": 50
      },
      "action": {
        "type": "expire"
      }
    },
    {
      "rulePriority": 2,
      "description": "SHA タグは30日で削除",
      "selection": {
        "tagStatus": "tagged",
        "tagPrefixList": ["sha-"],
        "countType": "sinceImagePushed",
        "countUnit": "days",
        "countNumber": 30
      },
      "action": {
        "type": "expire"
      }
    },
    {
      "rulePriority": 3,
      "description": "タグなしイメージは7日で削除",
      "selection": {
        "tagStatus": "untagged",
        "countType": "sinceImagePushed",
        "countUnit": "days",
        "countNumber": 7
      },
      "action": {
        "type": "expire"
      }
    }
  ]
}
```

```bash
# ECR リポジトリ管理コマンド集

# リポジトリの作成
aws ecr create-repository \
  --repository-name myorg/myapp \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration encryptionType=AES256 \
  --region ap-northeast-1

# イメージスキャン結果の確認
aws ecr describe-image-scan-findings \
  --repository-name myorg/myapp \
  --image-id imageTag=v1.2.3 \
  --region ap-northeast-1

# ライフサイクルポリシーの適用
aws ecr put-lifecycle-policy \
  --repository-name myorg/myapp \
  --lifecycle-policy-text file://ecr-lifecycle-policy.json

# クロスアカウントアクセスの設定
aws ecr set-repository-policy \
  --repository-name myorg/myapp \
  --policy-text '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Sid": "AllowPull",
        "Effect": "Allow",
        "Principal": {"AWS": "arn:aws:iam::987654321098:root"},
        "Action": [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
      }
    ]
  }'
```

### 3.2 GHCR (GitHub Container Registry) の管理

```yaml
# .github/workflows/cleanup-ghcr.yml — 古いイメージの自動削除
name: Cleanup GHCR Images

on:
  schedule:
    - cron: '0 3 * * 0'  # 毎週日曜 3:00 UTC

jobs:
  cleanup:
    runs-on: ubuntu-latest
    permissions:
      packages: write

    steps:
      - name: Delete old untagged images
        uses: actions/delete-package-versions@v5
        with:
          package-name: myapp
          package-type: container
          min-versions-to-keep: 10
          delete-only-untagged-versions: true

      - name: Delete old pre-release images
        uses: actions/delete-package-versions@v5
        with:
          package-name: myapp
          package-type: container
          min-versions-to-keep: 5
          ignore-versions: '^v\\d+\\.\\d+\\.\\d+$'
```

---

## 4. GitHub Actions — イメージビルドとプッシュ

### 4.1 基本的なビルド・プッシュワークフロー

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

### 4.2 セキュリティスキャン統合ビルドワークフロー

```yaml
# .github/workflows/build-scan-push.yml
name: Build, Scan and Push

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      security-events: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build image for scanning
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          load: true
          tags: scan-target:latest
          cache-from: type=gha

      # Trivy による脆弱性スキャン
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: scan-target:latest
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'  # CRITICAL/HIGH があればジョブ失敗

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

      # Hadolint による Dockerfile リンティング
      - name: Run Hadolint
        uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
          failure-threshold: warning

      # Docker Scout による SBOM 生成
      - name: Docker Scout SBOM
        if: github.event_name != 'pull_request'
        uses: docker/scout-action@v1
        with:
          command: cves
          image: scan-target:latest
          sarif-file: scout-results.sarif

      # スキャン通過後にプッシュ
      - name: Login to Container Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        if: github.event_name != 'pull_request'
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=ref,event=tag
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Push to registry
        if: github.event_name != 'pull_request'
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
          provenance: true
          sbom: true
```

### 4.3 マルチ環境プロモーションワークフロー

```yaml
# .github/workflows/promote-image.yml
name: Promote Image to Production

on:
  workflow_dispatch:
    inputs:
      image-tag:
        description: 'Image tag to promote (e.g., sha-abc1234)'
        required: true
      target-env:
        description: 'Target environment'
        required: true
        type: choice
        options:
          - staging
          - production

jobs:
  promote:
    runs-on: ubuntu-latest
    environment: ${{ inputs.target-env }}
    permissions:
      packages: write
      contents: write

    steps:
      - uses: actions/checkout@v4

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Promote image
        run: |
          SOURCE="ghcr.io/${{ github.repository }}:${{ inputs.image-tag }}"
          TARGET="ghcr.io/${{ github.repository }}:${{ inputs.target-env }}-latest"

          docker pull "$SOURCE"
          docker tag "$SOURCE" "$TARGET"
          docker push "$TARGET"

          echo "Promoted $SOURCE -> $TARGET"

      - name: Update deployment manifest
        if: inputs.target-env == 'production'
        run: |
          # K8s マニフェストのイメージタグを更新
          sed -i "s|image: ghcr.io/${{ github.repository }}:.*|image: ghcr.io/${{ github.repository }}:${{ inputs.image-tag }}|" \
            k8s/overlays/production/deployment-patch.yaml

          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add k8s/overlays/production/deployment-patch.yaml
          git commit -m "chore(deploy): promote ${{ inputs.image-tag }} to production"
          git push
```

---

## 5. ECS (Fargate) デプロイ

### 5.1 タスク定義

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

### 5.2 ECS サービス定義 (Terraform)

```hcl
# ecs-service.tf — ECS サービスの Terraform 定義

resource "aws_ecs_cluster" "main" {
  name = "myapp-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  configuration {
    execute_command_configuration {
      logging = "OVERRIDE"
      log_configuration {
        cloud_watch_log_group_name = aws_cloudwatch_log_group.ecs_exec.name
      }
    }
  }
}

resource "aws_ecs_service" "myapp" {
  name            = "myapp-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.myapp.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  # デプロイ設定
  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 100
  health_check_grace_period_seconds  = 60

  # デプロイサーキットブレーカー
  deployment_circuit_breaker {
    enable   = true
    rollback = true  # 失敗時に自動ロールバック
  }

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.myapp.arn
    container_name   = "app"
    container_port   = 3000
  }

  # サービスディスカバリ
  service_registries {
    registry_arn = aws_service_discovery_service.myapp.arn
  }

  lifecycle {
    ignore_changes = [task_definition]  # CI/CD でタスク定義を更新
  }
}

# オートスケーリング設定
resource "aws_appautoscaling_target" "myapp" {
  max_capacity       = 10
  min_capacity       = 3
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.myapp.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "myapp-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.myapp.resource_id
  scalable_dimension = aws_appautoscaling_target.myapp.scalable_dimension
  service_namespace  = aws_appautoscaling_target.myapp.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

resource "aws_appautoscaling_policy" "memory" {
  name               = "myapp-memory-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.myapp.resource_id
  scalable_dimension = aws_appautoscaling_target.myapp.scalable_dimension
  service_namespace  = aws_appautoscaling_target.myapp.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    target_value       = 80.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}
```

### 5.3 ECS デプロイワークフロー

```yaml
# .github/workflows/deploy-ecs.yml
name: Deploy to ECS

on:
  workflow_run:
    workflows: ["Build and Push Container Image"]
    types: [completed]
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions-deploy
          aws-region: ap-northeast-1

      - name: Get image tag
        id: image
        run: |
          echo "tag=sha-${GITHUB_SHA::7}" >> $GITHUB_OUTPUT

      - name: Update ECS task definition
        id: task-def
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: ecs-task-definition.json
          container-name: app
          image: ghcr.io/${{ github.repository }}:${{ steps.image.outputs.tag }}

      - name: Deploy to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v1
        with:
          task-definition: ${{ steps.task-def.outputs.task-definition }}
          service: myapp-service
          cluster: myapp-cluster
          wait-for-service-stability: true
          wait-for-minutes: 10

      - name: Notify deployment result
        if: always()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "${{ job.status == 'success' && 'ECS deploy succeeded' || 'ECS deploy FAILED' }}: ${{ steps.image.outputs.tag }}"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
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

## 6. Kubernetes + ArgoCD (GitOps)

### 6.1 Kubernetes マニフェスト (Kustomize ベース)

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
          startupProbe:
            httpGet:
              path: /health
              port: 3000
            failureThreshold: 30
            periodSeconds: 2
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
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

### 6.2 Kustomize による環境差分管理

```yaml
# k8s/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - service.yaml
  - hpa.yaml
  - pdb.yaml

commonLabels:
  app.kubernetes.io/name: myapp
  app.kubernetes.io/managed-by: kustomize
```

```yaml
# k8s/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: production

resources:
  - ../../base

patches:
  - path: deployment-patch.yaml
  - path: hpa-patch.yaml

configMapGenerator:
  - name: myapp-config
    literals:
      - LOG_LEVEL=info
      - CACHE_TTL=3600

images:
  - name: ghcr.io/myorg/myapp
    newTag: v1.2.3
```

```yaml
# k8s/overlays/production/deployment-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 5
  template:
    spec:
      containers:
        - name: app
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: 1000m
              memory: 1Gi
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: myapp
```

```yaml
# k8s/base/pdb.yaml — Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: myapp
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: myapp
```

### 6.3 ArgoCD Application 定義

```yaml
# argocd/application.yaml — ArgoCD Application 定義
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-production
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
  annotations:
    notifications.argoproj.io/subscribe.on-sync-succeeded.slack: deployments
    notifications.argoproj.io/subscribe.on-sync-failed.slack: deployments-alerts
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
      - PrunePropagationPolicy=foreground
      - PruneLast=true
    retry:
      limit: 3
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
  ignoreDifferences:
    - group: apps
      kind: Deployment
      jsonPointers:
        - /spec/replicas  # HPA が管理するためArgoCD で無視
```

### 6.4 ArgoCD ApplicationSet によるマルチ環境管理

```yaml
# argocd/applicationset.yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: myapp-environments
  namespace: argocd
spec:
  generators:
    - list:
        elements:
          - env: development
            cluster: https://dev-cluster.example.com
            revision: develop
            autoSync: true
          - env: staging
            cluster: https://staging-cluster.example.com
            revision: main
            autoSync: true
          - env: production
            cluster: https://prod-cluster.example.com
            revision: main
            autoSync: false  # 本番は手動同期
  template:
    metadata:
      name: 'myapp-{{env}}'
      namespace: argocd
    spec:
      project: default
      source:
        repoURL: https://github.com/myorg/myapp-k8s-manifests.git
        targetRevision: '{{revision}}'
        path: 'k8s/overlays/{{env}}'
      destination:
        server: '{{cluster}}'
        namespace: '{{env}}'
      syncPolicy:
        automated:
          prune: '{{autoSync}}'
          selfHeal: '{{autoSync}}'
```

### 6.5 Argo Rollouts による Progressive Delivery

```yaml
# k8s/base/rollout.yaml — Canary デプロイ
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: myapp
spec:
  replicas: 5
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
        - name: app
          image: ghcr.io/myorg/myapp:v1.2.3
          ports:
            - containerPort: 3000
  strategy:
    canary:
      canaryService: myapp-canary
      stableService: myapp-stable
      trafficRouting:
        nginx:
          stableIngress: myapp-ingress
      steps:
        - setWeight: 5        # 5% のトラフィックを新バージョンへ
        - pause: { duration: 5m }
        - analysis:            # メトリクスベースの自動判定
            templates:
              - templateName: success-rate
            args:
              - name: service-name
                value: myapp-canary
        - setWeight: 25
        - pause: { duration: 5m }
        - analysis:
            templates:
              - templateName: success-rate
        - setWeight: 50
        - pause: { duration: 10m }
        - setWeight: 100
      analysis:
        successfulRunHistoryLimit: 3
        unsuccessfulRunHistoryLimit: 3
---
# k8s/base/analysis-template.yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  args:
    - name: service-name
  metrics:
    - name: success-rate
      interval: 60s
      successCondition: result[0] >= 0.99
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            sum(rate(http_requests_total{service="{{args.service-name}}",status!~"5.."}[5m]))
            /
            sum(rate(http_requests_total{service="{{args.service-name}}"}[5m]))
```

---

## 7. コンテナネットワーキングとサービスメッシュ

### 7.1 Istio によるサービスメッシュ構成

```yaml
# istio/virtual-service.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: myapp
spec:
  hosts:
    - myapp.example.com
  gateways:
    - myapp-gateway
  http:
    - match:
        - headers:
            x-canary:
              exact: "true"
      route:
        - destination:
            host: myapp
            subset: canary
    - route:
        - destination:
            host: myapp
            subset: stable
          weight: 95
        - destination:
            host: myapp
            subset: canary
          weight: 5
      timeout: 30s
      retries:
        attempts: 3
        perTryTimeout: 10s
        retryOn: 5xx,reset,connect-failure
---
# istio/destination-rule.yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: myapp
spec:
  host: myapp
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        h2UpgradePolicy: DEFAULT
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
    circuitBreaker:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
  subsets:
    - name: stable
      labels:
        version: stable
    - name: canary
      labels:
        version: canary
```

---

## 8. 比較表

| 特性 | ECS (Fargate) | Kubernetes (EKS) | Docker Compose |
|------|--------------|-------------------|----------------|
| 管理負荷 | 低い | 高い | 最低 |
| スケーラビリティ | 高い | 最高 | 低い |
| 学習コスト | 中 | 高い | 低い |
| エコシステム | AWS 内完結 | 巨大 (CNCF) | 限定的 |
| コスト | 中 | 高い (コントロールプレーン有料) | 低い |
| GitOps 対応 | CodePipeline | ArgoCD / Flux | 困難 |
| 適用規模 | 中〜大規模 | 大規模 | 開発/小規模 |
| サービスメッシュ | App Mesh | Istio / Linkerd | なし |
| マルチクラウド | 不可 | 可能 | 不可 |

| GitOps ツール比較 | ArgoCD | Flux | Jenkins X |
|-------------------|--------|------|-----------|
| UI ダッシュボード | 充実 | 基本的 | あり |
| マルチクラスタ | 対応 | 対応 | 限定的 |
| Helm 対応 | 対応 | 対応 | 対応 |
| Kustomize 対応 | 対応 | 対応 | 限定的 |
| RBAC | 細かい制御 | K8s RBAC | 独自 |
| コミュニティ | 大きい | 大きい | 小さい |
| Progressive Delivery | Argo Rollouts | Flagger | 限定的 |
| ApplicationSet | 対応 | Kustomization | 非対応 |

| コンテナレジストリ比較 | ECR | GHCR | Docker Hub | Harbor |
|----------------------|-----|------|------------|--------|
| 運用形態 | AWS マネージド | GitHub マネージド | SaaS | セルフホスト |
| スキャン機能 | あり | なし (外部連携) | 有料プラン | あり |
| ライフサイクルポリシー | あり | 手動/Actions | なし | あり |
| マルチアーキテクチャ | 対応 | 対応 | 対応 | 対応 |
| プライベートリポジトリ | 無制限 | 無制限 | 1つ (無料) | 無制限 |
| コスト | ストレージ + 転送量 | 無料枠あり | 無料枠あり | インフラ費 |

---

## 9. コンテナランタイムセキュリティ

### 9.1 Pod セキュリティスタンダード

```yaml
# k8s/base/pod-security.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
---
# セキュアな Pod 設定の例
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1001
    fsGroup: 1001
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: ghcr.io/myorg/myapp:v1.2.3
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop:
            - ALL
      volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/.cache
  volumes:
    - name: tmp
      emptyDir: {}
    - name: cache
      emptyDir:
        sizeLimit: 100Mi
```

### 9.2 NetworkPolicy による通信制御

```yaml
# k8s/base/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: myapp-network-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # ALB/Ingress Controller からのみ受信許可
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
      ports:
        - port: 3000
          protocol: TCP
  egress:
    # DNS 解決
    - to: []
      ports:
        - port: 53
          protocol: UDP
        - port: 53
          protocol: TCP
    # データベース
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: database
      ports:
        - port: 5432
          protocol: TCP
    # 外部 HTTPS API
    - to: []
      ports:
        - port: 443
          protocol: TCP
```

---

## 10. 運用コマンド集

### 10.1 ECS 運用コマンド

```bash
# タスクのリスト表示
aws ecs list-tasks \
  --cluster myapp-cluster \
  --service-name myapp-service

# タスクの詳細確認
aws ecs describe-tasks \
  --cluster myapp-cluster \
  --tasks arn:aws:ecs:ap-northeast-1:123456789012:task/myapp-cluster/abc123

# サービスのイベントログ確認
aws ecs describe-services \
  --cluster myapp-cluster \
  --services myapp-service \
  --query 'services[0].events[:10]' \
  --output table

# 手動スケーリング
aws ecs update-service \
  --cluster myapp-cluster \
  --service myapp-service \
  --desired-count 5

# ECS Exec でコンテナに接続 (デバッグ用)
aws ecs execute-command \
  --cluster myapp-cluster \
  --task abc123 \
  --container app \
  --interactive \
  --command "/bin/sh"

# 強制的な新デプロイ
aws ecs update-service \
  --cluster myapp-cluster \
  --service myapp-service \
  --force-new-deployment
```

### 10.2 Kubernetes 運用コマンド

```bash
# Pod のステータス確認
kubectl get pods -n production -l app=myapp -o wide

# Pod のログ確認 (全 Pod のログを追跡)
kubectl logs -n production -l app=myapp --all-containers --follow --tail=100

# Pod の詳細情報
kubectl describe pod -n production <pod-name>

# ローリングアップデートの状況確認
kubectl rollout status deployment/myapp -n production

# ロールバック
kubectl rollout undo deployment/myapp -n production

# 特定リビジョンへのロールバック
kubectl rollout undo deployment/myapp -n production --to-revision=3

# リビジョン履歴の確認
kubectl rollout history deployment/myapp -n production

# Pod へのポートフォワード (デバッグ用)
kubectl port-forward -n production svc/myapp 8080:80

# リソース使用量の確認
kubectl top pods -n production -l app=myapp

# HPA のステータス
kubectl get hpa -n production myapp -o yaml

# ArgoCD の同期ステータス
argocd app get myapp-production
argocd app sync myapp-production
argocd app diff myapp-production

# ArgoCD のロールバック
argocd app rollback myapp-production
```

---

## 11. アンチパターン

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

### アンチパターン 3: シークレットのハードコーディング

```yaml
# 悪い例: 環境変数にシークレットを直接記載
containers:
  - name: app
    env:
      - name: DATABASE_URL
        value: "postgresql://user:password123@db:5432/mydb"  # 危険!

# 良い例: Kubernetes Secret または外部シークレット管理
containers:
  - name: app
    env:
      - name: DATABASE_URL
        valueFrom:
          secretKeyRef:
            name: myapp-secrets
            key: database-url

# さらに良い例: External Secrets Operator で AWS Secrets Manager と連携
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: myapp-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: myapp-secrets
  data:
    - secretKey: database-url
      remoteRef:
        key: myapp/production/database-url
```

### アンチパターン 4: ヘルスチェック未設定

```yaml
# 悪い例: ヘルスチェックなし
containers:
  - name: app
    image: myapp:v1.0.0
    ports:
      - containerPort: 3000
    # アプリがハングしてもトラフィックが流れ続ける

# 良い例: 3種類のプローブを適切に設定
containers:
  - name: app
    image: myapp:v1.0.0
    # startupProbe: 起動完了を判定 (初回のみ)
    startupProbe:
      httpGet:
        path: /health
        port: 3000
      failureThreshold: 30
      periodSeconds: 2
    # readinessProbe: トラフィック受信可能かを判定
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 3000
      periodSeconds: 10
      failureThreshold: 3
    # livenessProbe: アプリが生存しているかを判定
    livenessProbe:
      httpGet:
        path: /health/live
        port: 3000
      periodSeconds: 30
      failureThreshold: 3
```

### アンチパターン 5: 単一レプリカの本番運用

```yaml
# 悪い例: レプリカ1で本番運用
spec:
  replicas: 1
  # Pod 再起動時にサービス断が発生

# 良い例: 最低3レプリカ + PDB + TopologySpreadConstraints
spec:
  replicas: 3
  template:
    spec:
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: myapp
# + PDB で最低2 Pod を保証
```

---

## 12. FAQ

### Q1: ECS と Kubernetes、どちらを選ぶべきですか？

チームの Kubernetes 経験と運用負荷の許容度で判断します。AWS に閉じたシンプルなコンテナ運用なら ECS Fargate が管理負荷が低く始めやすいです。マルチクラウド、高度なトラフィック制御（Istio 等）、豊富な OSS エコシステムが必要なら Kubernetes を選択してください。EKS のコントロールプレーン費用（約$73/月）も考慮に入れましょう。具体的な判断基準は以下の通りです:

- **ECS を選ぶべき場合**: AWS 専用環境、チームに Kubernetes 経験者が少ない、運用チームが小規模（1-3名）、サービス数が10未満
- **Kubernetes を選ぶべき場合**: マルチクラウド要件、サービスメッシュが必要、チームに Kubernetes 経験者がいる、サービス数が10以上、高度なスケジューリング要件

### Q2: ArgoCD の「自動同期」は常に有効にすべきですか？

開発環境では有効にして問題ありません。本番環境では `automated.prune: true` と `selfHeal: true` を慎重に検討してください。特に `prune` は Git リポジトリから削除されたリソースを自動削除するため、誤操作のリスクがあります。段階的に導入し、まずは手動同期（Sync ボタン）から始めて信頼性を確認することを推奨します。本番環境でのベストプラクティスは、`selfHeal: true`（手動変更の自動修正）は有効にしつつ、`prune: false`（自動削除は無効）とする構成です。

### Q3: コンテナイメージの脆弱性スキャンはどのタイミングで行うべきですか？

**ビルド時**（CI パイプライン内）と**定期スキャン**の2段階が推奨です。ビルド時には `trivy image` や `docker scout` を実行し、Critical/High の脆弱性があればビルドを失敗させます。レジストリに保存済みのイメージも、新しい CVE が公開される可能性があるため、日次の定期スキャンを設定してください。具体的には:

1. **PR 時**: Hadolint で Dockerfile のリンティング + Trivy でイメージスキャン
2. **マージ時**: フルスキャン + SBOM 生成 + イメージ署名（cosign）
3. **定期**: ECR のスキャン機能 or 日次の Trivy スキャン
4. **デプロイ前**: アドミッション制御で未スキャンイメージのデプロイを拒否

### Q4: Fargate と EC2 起動タイプ、どちらを使うべきですか？

**Fargate** はサーバー管理不要で運用コストが低いですが、GPU 対応が限定的で、カスタムカーネルパラメータの変更ができません。**EC2** は柔軟性が高く、スポットインスタンスでコスト最適化が可能ですが、EC2 インスタンスのパッチ適用やスケーリング管理が必要です。一般的な Web アプリケーションには Fargate を推奨し、GPU が必要な ML 推論ワークロードや特殊な要件がある場合に EC2 を選択してください。

### Q5: イメージプルに時間がかかる場合の対策は？

以下の対策を検討してください:

1. **イメージサイズの縮小**: マルチステージビルドと Alpine ベースイメージの使用
2. **イメージキャッシュ**: ECR Pull Through Cache を使用してクロスリージョンのプルを高速化
3. **レジストリの近接配置**: デプロイ先と同じリージョンにレジストリを配置
4. **Lazy Loading**: containerd の nerdctl + stargz で、イメージ全体をプルせずに起動（Seekable OCI）
5. **ウォームプール**: ECS Capacity Provider のウォームプールで事前にインスタンスを起動

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
| Kustomize | base + overlays で環境差分を管理。DRY なマニフェスト構成 |
| Progressive Delivery | Argo Rollouts + AnalysisTemplate で安全なカナリーデプロイ |
| セキュリティ | 非root 実行、readOnlyRootFilesystem、NetworkPolicy、イメージスキャン |
| サービスメッシュ | Istio/Linkerd で mTLS、トラフィック制御、可観測性を統合 |

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
5. **Argo Rollouts Documentation** — https://argoproj.github.io/argo-rollouts/ — Progressive Delivery の公式ガイド
6. **Istio Documentation** — https://istio.io/latest/docs/ — サービスメッシュの公式ドキュメント
7. **Kustomize Documentation** — https://kustomize.io/ — Kubernetes ネイティブの構成管理ツール
8. **Trivy Documentation** — https://aquasecurity.github.io/trivy/ — コンテナ脆弱性スキャナーの公式ドキュメント

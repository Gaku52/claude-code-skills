# クラウドデプロイ

> AWS、Vercel、Cloudflare Workers への実践的なデプロイ手法を習得し、プロジェクト特性に応じた最適なプラットフォームを選択する

## この章で学ぶこと

1. **AWS (ECS/Lambda/S3+CloudFront) へのデプロイ** — IaC を活用した本格的なクラウドインフラ構築とデプロイ自動化
2. **Vercel/Netlify によるフロントエンドデプロイ** — Git 連携による自動デプロイとプレビュー環境の活用
3. **Cloudflare Workers によるエッジデプロイ** — エッジコンピューティングの特性を活かしたサーバーレスデプロイ
4. **AWS ECS/Fargate によるコンテナデプロイ** — Docker コンテナを活用したスケーラブルなアプリケーション運用
5. **GCP Cloud Run / Firebase Hosting** — Google Cloud のマネージドサービスを利用した効率的なデプロイ
6. **マルチクラウド戦略** — 複数のクラウドプロバイダを組み合わせた最適なアーキテクチャ設計

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

### 1.1 プラットフォーム選択のデシジョンツリー

```
プロジェクト要件を確認
│
├── フロントエンドのみ（SSG/SSR）？
│   ├── Next.js → Vercel（最適化済み）
│   ├── Astro/Gatsby → Netlify or Cloudflare Pages
│   └── SPA (React/Vue) → S3+CloudFront or Cloudflare Pages
│
├── API バックエンドが必要？
│   ├── リクエスト駆動型（軽量）→ Lambda or Workers
│   ├── 常時稼働型（WebSocket等）→ ECS/Fargate or Cloud Run
│   └── ステートフル → ECS/EKS + EBS/EFS
│
├── エッジ処理が必要？
│   ├── A/B テスト → Cloudflare Workers or Lambda@Edge
│   ├── 地理的ルーティング → CloudFront Functions or Workers
│   └── リアルタイム変換 → Workers（Streams API）
│
└── VPC 内リソースへのアクセスが必要？
    ├── RDS/ElastiCache → Lambda (VPC) or ECS
    └── オンプレミス連携 → ECS + VPN/Direct Connect
```

### 1.2 デプロイの成熟度モデル

```
Level 0: 手動デプロイ
  └── FTP/SCP でファイルを直接配置、サーバーに SSH して操作

Level 1: スクリプトベース
  └── デプロイスクリプト（シェルスクリプト/Makefile）で半自動化

Level 2: CI/CD パイプライン
  └── GitHub Actions/Jenkins で自動ビルド＆デプロイ

Level 3: IaC + GitOps
  └── Terraform/CDK でインフラ定義、Git 操作でデプロイトリガー

Level 4: プログレッシブデリバリー
  └── Canary/Blue-Green + 自動ロールバック + 観測性統合
```

---

## 2. AWS デプロイ — S3 + CloudFront (静的サイト)

### 2.1 基本デプロイワークフロー

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

### 2.2 Terraform による S3 + CloudFront インフラ定義

```hcl
# terraform/modules/static-site/main.tf
# S3 バケット（静的サイトホスティング用）
resource "aws_s3_bucket" "site" {
  bucket = "${var.project_name}-${var.environment}-site"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_s3_bucket_versioning" "site" {
  bucket = aws_s3_bucket.site.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 バケットポリシー（CloudFront OAC からのアクセスのみ許可）
resource "aws_s3_bucket_policy" "site" {
  bucket = aws_s3_bucket.site.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowCloudFrontOAC"
        Effect    = "Allow"
        Principal = { Service = "cloudfront.amazonaws.com" }
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.site.arn}/*"
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = aws_cloudfront_distribution.site.arn
          }
        }
      }
    ]
  })
}

# CloudFront Origin Access Control
resource "aws_cloudfront_origin_access_control" "site" {
  name                              = "${var.project_name}-${var.environment}-oac"
  description                       = "OAC for ${var.project_name}"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# CloudFront ディストリビューション
resource "aws_cloudfront_distribution" "site" {
  origin {
    domain_name              = aws_s3_bucket.site.bucket_regional_domain_name
    origin_id                = "S3-${aws_s3_bucket.site.id}"
    origin_access_control_id = aws_cloudfront_origin_access_control.site.id
  }

  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  price_class         = "PriceClass_200"  # 北米+欧州+アジア

  aliases = var.domain_names

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3-${aws_s3_bucket.site.id}"
    viewer_protocol_policy = "redirect-to-https"
    compress               = true

    # マネージドキャッシュポリシー: CachingOptimized
    cache_policy_id = "658327ea-f89d-4fab-a63d-7e88639e58f6"

    # レスポンスヘッダーポリシー: SecurityHeadersPolicy
    response_headers_policy_id = "67f7725c-6f97-4210-82d7-5512b31e9d03"
  }

  # SPA 用: 404 を index.html にフォールバック
  custom_error_response {
    error_code            = 404
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 10
  }

  custom_error_response {
    error_code            = 403
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 10
  }

  viewer_certificate {
    acm_certificate_arn      = var.acm_certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Route 53 レコード
resource "aws_route53_record" "site" {
  for_each = toset(var.domain_names)

  zone_id = var.hosted_zone_id
  name    = each.value
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.site.domain_name
    zone_id                = aws_cloudfront_distribution.site.hosted_zone_id
    evaluate_target_health = false
  }
}

# 出力
output "cloudfront_distribution_id" {
  value = aws_cloudfront_distribution.site.id
}

output "cloudfront_domain_name" {
  value = aws_cloudfront_distribution.site.domain_name
}

output "s3_bucket_name" {
  value = aws_s3_bucket.site.id
}
```

### 2.3 高度なキャッシュ戦略

```yaml
# GitHub Actions — アセット種別ごとのキャッシュ制御付きデプロイ
- name: Deploy with granular cache control
  run: |
    # JavaScript/CSS（ハッシュ付きファイル名）: 1年キャッシュ
    aws s3 sync dist/assets/ s3://$BUCKET/assets/ \
      --delete \
      --cache-control "public, max-age=31536000, immutable" \
      --content-encoding gzip

    # 画像ファイル: 1ヶ月キャッシュ
    aws s3 sync dist/images/ s3://$BUCKET/images/ \
      --delete \
      --cache-control "public, max-age=2592000"

    # フォントファイル: 1年キャッシュ（CORS ヘッダー付き）
    aws s3 sync dist/fonts/ s3://$BUCKET/fonts/ \
      --delete \
      --cache-control "public, max-age=31536000, immutable" \
      --content-type "font/woff2"

    # HTML ファイル: キャッシュなし（常に最新を取得）
    find dist/ -name "*.html" -exec \
      aws s3 cp {} s3://$BUCKET/{} \
        --cache-control "no-cache, no-store, must-revalidate" \;

    # Service Worker: キャッシュなし
    aws s3 cp dist/sw.js s3://$BUCKET/sw.js \
      --cache-control "no-cache, no-store, must-revalidate"

    # manifest.json: 短期キャッシュ
    aws s3 cp dist/manifest.json s3://$BUCKET/manifest.json \
      --cache-control "public, max-age=3600"
```

---

## 3. AWS Lambda デプロイ (SAM)

### 3.1 SAM テンプレート

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

### 3.2 SAM デプロイ GitHub Actions ワークフロー

```yaml
# .github/workflows/deploy-sam.yml
name: Deploy SAM Application

on:
  push:
    branches: [main]
    paths:
      - 'backend/**'
      - 'template.yaml'
      - '.github/workflows/deploy-sam.yml'

concurrency:
  group: deploy-sam-${{ github.ref }}
  cancel-in-progress: false  # デプロイは途中キャンセルしない

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm test

  deploy:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    environment:
      name: production
      url: ${{ steps.deploy.outputs.api_url }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - name: Install and Build
        run: |
          npm ci
          npm run build

      - name: Setup SAM CLI
        uses: aws-actions/setup-sam@v2
        with:
          use-installer: true

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ap-northeast-1

      - name: SAM Build
        run: sam build --use-container

      - name: SAM Deploy
        id: deploy
        run: |
          sam deploy \
            --stack-name my-app-prod \
            --s3-bucket ${{ secrets.SAM_ARTIFACT_BUCKET }} \
            --capabilities CAPABILITY_IAM \
            --no-confirm-changeset \
            --no-fail-on-empty-changeset \
            --parameter-overrides \
              Environment=production \
              DatabaseHost=${{ secrets.DB_HOST }}

          # デプロイ後の API URL を取得
          API_URL=$(aws cloudformation describe-stacks \
            --stack-name my-app-prod \
            --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
            --output text)
          echo "api_url=${API_URL}" >> "$GITHUB_OUTPUT"

      - name: Smoke Test
        run: |
          API_URL="${{ steps.deploy.outputs.api_url }}"
          STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/health")
          if [ "$STATUS" != "200" ]; then
            echo "Smoke test failed with status: $STATUS"
            exit 1
          fi
          echo "Smoke test passed: ${API_URL}/health returned 200"
```

### 3.3 Lambda レイヤーの活用

```yaml
# template.yaml — Lambda Layers を使った依存関係の分離
Resources:
  SharedDependenciesLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: shared-dependencies
      Description: Shared npm dependencies
      ContentUri: layers/shared/
      CompatibleRuntimes:
        - nodejs20.x
      RetentionPolicy: Retain
    Metadata:
      BuildMethod: nodejs20.x

  ApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: dist/api.handler
      Layers:
        - !Ref SharedDependenciesLayer
      # ...

  WorkerFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: dist/worker.handler
      Layers:
        - !Ref SharedDependenciesLayer
      Events:
        SQSEvent:
          Type: SQS
          Properties:
            Queue: !GetAtt TaskQueue.Arn
            BatchSize: 10
            MaximumBatchingWindowInSeconds: 5
```

---

## 4. AWS ECS/Fargate デプロイ

### 4.1 ECS タスク定義と GitHub Actions ワークフロー

```yaml
# .github/workflows/deploy-ecs.yml
name: Deploy to ECS Fargate

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'Dockerfile'
      - '.github/workflows/deploy-ecs.yml'

env:
  AWS_REGION: ap-northeast-1
  ECR_REPOSITORY: my-app
  ECS_CLUSTER: my-app-cluster
  ECS_SERVICE: my-app-service
  TASK_DEFINITION: .aws/task-definition.json
  CONTAINER_NAME: my-app

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, tag, and push Docker image
        id: build-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build \
            --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
            --build-arg GIT_SHA=${{ github.sha }} \
            -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
            -t $ECR_REGISTRY/$ECR_REPOSITORY:latest \
            .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> "$GITHUB_OUTPUT"

      - name: Download current task definition
        run: |
          aws ecs describe-task-definition \
            --task-definition ${{ env.ECS_SERVICE }} \
            --query 'taskDefinition' \
            --output json > task-definition.json

      - name: Update task definition with new image
        id: task-def
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: task-definition.json
          container-name: ${{ env.CONTAINER_NAME }}
          image: ${{ steps.build-image.outputs.image }}

      - name: Deploy to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition: ${{ steps.task-def.outputs.task-definition }}
          service: ${{ env.ECS_SERVICE }}
          cluster: ${{ env.ECS_CLUSTER }}
          wait-for-service-stability: true
          wait-for-minutes: 10

      - name: Verify deployment
        run: |
          RUNNING_TASKS=$(aws ecs list-tasks \
            --cluster $ECS_CLUSTER \
            --service-name $ECS_SERVICE \
            --desired-status RUNNING \
            --query 'taskArns | length(@)')
          echo "Running tasks: $RUNNING_TASKS"

          TASK_ARN=$(aws ecs list-tasks \
            --cluster $ECS_CLUSTER \
            --service-name $ECS_SERVICE \
            --desired-status RUNNING \
            --query 'taskArns[0]' \
            --output text)

          TASK_IMAGE=$(aws ecs describe-tasks \
            --cluster $ECS_CLUSTER \
            --tasks $TASK_ARN \
            --query "tasks[0].containers[?name=='$CONTAINER_NAME'].image" \
            --output text)

          echo "Deployed image: $TASK_IMAGE"
          echo "Expected image: ${{ steps.build-image.outputs.image }}"

          if [ "$TASK_IMAGE" != "${{ steps.build-image.outputs.image }}" ]; then
            echo "Image mismatch! Deployment verification failed."
            exit 1
          fi
```

### 4.2 ECS タスク定義（Terraform）

```hcl
# terraform/modules/ecs-service/main.tf
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-${var.environment}"

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

resource "aws_ecs_task_definition" "app" {
  family                   = "${var.project_name}-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = var.container_name
      image = "${var.ecr_repository_url}:latest"
      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port
          protocol      = "tcp"
        }
      ]
      environment = [
        { name = "NODE_ENV", value = var.environment },
        { name = "PORT", value = tostring(var.container_port) },
      ]
      secrets = [
        {
          name      = "DATABASE_URL"
          valueFrom = "${var.secrets_arn}:DATABASE_URL::"
        },
        {
          name      = "API_KEY"
          valueFrom = "${var.secrets_arn}:API_KEY::"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.container_port}/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

resource "aws_ecs_service" "app" {
  name            = "${var.project_name}-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  health_check_grace_period_seconds  = 60

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = var.container_name
    container_port   = var.container_port
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true  # デプロイ失敗時に自動ロールバック
  }

  lifecycle {
    ignore_changes = [task_definition]  # CI/CD で更新するため
  }
}

# Auto Scaling
resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${var.project_name}-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}
```

---

## 5. Vercel デプロイ

### 5.1 Vercel 設定

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

### 5.2 Vercel + GitHub Actions 連携（カスタムパイプライン）

```yaml
# .github/workflows/deploy-vercel.yml
name: Deploy to Vercel with Custom Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
  VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm run lint
      - run: npm run type-check
      - run: npm test -- --coverage

  lighthouse:
    needs: test
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci

      - name: Pull Vercel Preview
        run: |
          npx vercel pull --yes --environment=preview --token=${{ secrets.VERCEL_TOKEN }}
          npx vercel build --token=${{ secrets.VERCEL_TOKEN }}
          PREVIEW_URL=$(npx vercel deploy --prebuilt --token=${{ secrets.VERCEL_TOKEN }})
          echo "PREVIEW_URL=${PREVIEW_URL}" >> "$GITHUB_ENV"

      - name: Run Lighthouse CI
        uses: treosh/lighthouse-ci-action@v12
        with:
          urls: |
            ${{ env.PREVIEW_URL }}
            ${{ env.PREVIEW_URL }}/about
          budgetPath: ./lighthouse-budget.json
          uploadArtifacts: true

  deploy-production:
    needs: test
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    environment:
      name: production
      url: ${{ steps.deploy.outputs.url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci

      - name: Pull Vercel Environment
        run: npx vercel pull --yes --environment=production --token=${{ secrets.VERCEL_TOKEN }}

      - name: Build
        run: npx vercel build --prod --token=${{ secrets.VERCEL_TOKEN }}

      - name: Deploy to Production
        id: deploy
        run: |
          URL=$(npx vercel deploy --prebuilt --prod --token=${{ secrets.VERCEL_TOKEN }})
          echo "url=${URL}" >> "$GITHUB_OUTPUT"
          echo "Deployed to: ${URL}"

      - name: Verify Deployment
        run: |
          sleep 10  # Edge Network の伝播を待機
          STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${{ steps.deploy.outputs.url }}")
          if [ "$STATUS" != "200" ]; then
            echo "Deployment verification failed: HTTP $STATUS"
            exit 1
          fi
```

### 5.3 Vercel Edge Functions

```typescript
// app/api/geo/route.ts — Vercel Edge Function
import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'edge';
export const preferredRegion = ['nrt1', 'iad1'];  // 東京 + バージニア

export async function GET(request: NextRequest) {
  const geo = request.geo;
  const ip = request.ip;

  // エッジでの地理情報に基づくレスポンス
  const response = {
    country: geo?.country ?? 'unknown',
    city: geo?.city ?? 'unknown',
    region: geo?.region ?? 'unknown',
    latitude: geo?.latitude,
    longitude: geo?.longitude,
    ip: ip,
    timestamp: new Date().toISOString(),
    edge_region: process.env.VERCEL_REGION,
  };

  return NextResponse.json(response, {
    headers: {
      'Cache-Control': 's-maxage=60, stale-while-revalidate=300',
    },
  });
}
```

```typescript
// middleware.ts — Vercel Edge Middleware（全リクエストに適用）
import { NextRequest, NextResponse } from 'next/server';

export function middleware(request: NextRequest) {
  const url = request.nextUrl.clone();
  const country = request.geo?.country;

  // 地域別リダイレクト
  if (country === 'JP' && !url.pathname.startsWith('/ja')) {
    url.pathname = `/ja${url.pathname}`;
    return NextResponse.redirect(url);
  }

  // A/B テスト: Cookie ベースのバケット割り当て
  const bucket = request.cookies.get('ab-bucket')?.value;
  if (!bucket) {
    const newBucket = Math.random() < 0.5 ? 'control' : 'variant';
    const response = NextResponse.next();
    response.cookies.set('ab-bucket', newBucket, {
      maxAge: 60 * 60 * 24 * 30,  // 30日
      httpOnly: true,
      sameSite: 'lax',
    });
    return response;
  }

  // レート制限ヘッダー
  const response = NextResponse.next();
  response.headers.set('X-RateLimit-Limit', '100');
  response.headers.set('X-Robots-Tag', 'noindex, nofollow');

  return response;
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
```

---

## 6. Cloudflare Workers デプロイ

### 6.1 Worker 実装

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

### 6.2 Wrangler 設定

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

### 6.3 Cloudflare Workers GitHub Actions デプロイ

```yaml
# .github/workflows/deploy-workers.yml
name: Deploy to Cloudflare Workers

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'wrangler.toml'
      - '.github/workflows/deploy-workers.yml'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm test

      # Miniflare を使ったローカル統合テスト
      - name: Integration Test with Miniflare
        run: |
          npx wrangler dev --local --port 8787 &
          sleep 3

          # ヘルスチェック
          curl -f http://localhost:8787/api/health

          # API テスト
          RESPONSE=$(curl -s -X POST http://localhost:8787/api/items \
            -H 'Content-Type: application/json' \
            -d '{"name":"test","value":"data"}')
          echo "$RESPONSE" | jq -e '.success == true'

          kill %1

  deploy-staging:
    needs: test
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci

      - name: Deploy to Staging
        uses: cloudflare/wrangler-action@v3
        with:
          apiToken: ${{ secrets.CF_API_TOKEN }}
          accountId: ${{ secrets.CF_ACCOUNT_ID }}
          command: deploy --env staging

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://api.example.com
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci

      - name: Deploy to Production
        uses: cloudflare/wrangler-action@v3
        with:
          apiToken: ${{ secrets.CF_API_TOKEN }}
          accountId: ${{ secrets.CF_ACCOUNT_ID }}
          command: deploy --env production

      - name: Smoke Test
        run: |
          sleep 5
          STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.example.com/api/health)
          if [ "$STATUS" != "200" ]; then
            echo "Smoke test failed: HTTP $STATUS"
            exit 1
          fi
          echo "Smoke test passed"
```

### 6.4 Durable Objects を使ったステートフル Worker

```typescript
// src/counter.ts — Durable Object（ステートフルなエッジ処理）
export class Counter {
  private state: DurableObjectState;
  private env: Env;

  constructor(state: DurableObjectState, env: Env) {
    this.state = state;
    this.env = env;
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    switch (url.pathname) {
      case '/increment': {
        let count = (await this.state.storage.get<number>('count')) ?? 0;
        count += 1;
        await this.state.storage.put('count', count);
        return Response.json({ count });
      }
      case '/get': {
        const count = (await this.state.storage.get<number>('count')) ?? 0;
        return Response.json({ count });
      }
      case '/reset': {
        await this.state.storage.put('count', 0);
        return Response.json({ count: 0 });
      }
      default:
        return new Response('Not Found', { status: 404 });
    }
  }
}

// Worker からの Durable Object 呼び出し
async function handleCounter(
  request: Request,
  env: Env
): Promise<Response> {
  const url = new URL(request.url);
  const counterId = url.searchParams.get('id') ?? 'default';

  // Durable Object のスタブを取得
  const id = env.COUNTER.idFromName(counterId);
  const stub = env.COUNTER.get(id);

  // Durable Object にリクエストを転送
  return stub.fetch(request);
}
```

### 6.5 Cloudflare Pages + Functions

```typescript
// functions/api/[[route]].ts — Cloudflare Pages Functions
import { Hono } from 'hono';

type Bindings = {
  DB: D1Database;
  KV: KVNamespace;
};

const app = new Hono<{ Bindings: Bindings }>();

app.get('/api/posts', async (c) => {
  const { results } = await c.env.DB
    .prepare('SELECT * FROM posts ORDER BY created_at DESC LIMIT 20')
    .all();
  return c.json(results);
});

app.get('/api/posts/:id', async (c) => {
  const id = c.req.param('id');

  // KV キャッシュを確認
  const cached = await c.env.KV.get(`post:${id}`, 'json');
  if (cached) {
    return c.json(cached);
  }

  const post = await c.env.DB
    .prepare('SELECT * FROM posts WHERE id = ?')
    .bind(id)
    .first();

  if (!post) {
    return c.json({ error: 'Not Found' }, 404);
  }

  // KV にキャッシュ（60秒 TTL）
  c.executionCtx.waitUntil(
    c.env.KV.put(`post:${id}`, JSON.stringify(post), { expirationTtl: 60 })
  );

  return c.json(post);
});

app.post('/api/posts', async (c) => {
  const body = await c.req.json<{ title: string; content: string }>();

  const result = await c.env.DB
    .prepare('INSERT INTO posts (title, content) VALUES (?, ?) RETURNING *')
    .bind(body.title, body.content)
    .first();

  return c.json(result, 201);
});

export const onRequest = app.fetch;
```

---

## 7. GCP Cloud Run デプロイ

### 7.1 Cloud Run GitHub Actions ワークフロー

```yaml
# .github/workflows/deploy-cloud-run.yml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

env:
  PROJECT_ID: my-project-id
  REGION: asia-northeast1
  SERVICE_NAME: my-app
  IMAGE_NAME: asia-northeast1-docker.pkg.dev/my-project-id/my-repo/my-app

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Authenticate to Google Cloud (OIDC)
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Configure Docker for Artifact Registry
        run: gcloud auth configure-docker asia-northeast1-docker.pkg.dev

      - name: Build and Push Docker image
        run: |
          docker build \
            --build-arg GIT_SHA=${{ github.sha }} \
            -t $IMAGE_NAME:${{ github.sha }} \
            -t $IMAGE_NAME:latest \
            .
          docker push $IMAGE_NAME:${{ github.sha }}
          docker push $IMAGE_NAME:latest

      - name: Deploy to Cloud Run
        uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: ${{ env.SERVICE_NAME }}
          region: ${{ env.REGION }}
          image: ${{ env.IMAGE_NAME }}:${{ github.sha }}
          flags: |
            --memory=512Mi
            --cpu=1
            --min-instances=0
            --max-instances=10
            --concurrency=80
            --timeout=300
            --port=8080
            --cpu-throttling
            --session-affinity
          env_vars: |
            NODE_ENV=production
            GIT_SHA=${{ github.sha }}
          secrets: |
            DATABASE_URL=database-url:latest
            API_KEY=api-key:latest

      - name: Set traffic to new revision
        run: |
          # 段階的トラフィック移行（Canary）
          gcloud run services update-traffic $SERVICE_NAME \
            --region=$REGION \
            --to-revisions=LATEST=10

          # ヘルスチェック
          sleep 30
          SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
            --region=$REGION \
            --format='value(status.url)')
          STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/health")

          if [ "$STATUS" = "200" ]; then
            echo "Health check passed. Routing 100% traffic to new revision."
            gcloud run services update-traffic $SERVICE_NAME \
              --region=$REGION \
              --to-revisions=LATEST=100
          else
            echo "Health check failed. Rolling back."
            gcloud run services update-traffic $SERVICE_NAME \
              --region=$REGION \
              --to-revisions=LATEST=0
            exit 1
          fi
```

### 7.2 Firebase Hosting + Cloud Functions

```yaml
# .github/workflows/deploy-firebase.yml
name: Deploy to Firebase

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - name: Install dependencies
        run: |
          npm ci
          cd functions && npm ci

      - name: Build
        run: |
          npm run build
          cd functions && npm run build

      - name: Deploy to Firebase
        uses: FirebaseExtended/action-hosting-deploy@v0
        with:
          repoToken: ${{ secrets.GITHUB_TOKEN }}
          firebaseServiceAccount: ${{ secrets.FIREBASE_SERVICE_ACCOUNT }}
          channelId: live  # 本番チャネル
          projectId: my-project-id
```

```json
// firebase.json
{
  "hosting": {
    "public": "dist",
    "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
    "rewrites": [
      {
        "source": "/api/**",
        "function": "api"
      },
      {
        "source": "**",
        "destination": "/index.html"
      }
    ],
    "headers": [
      {
        "source": "/**/*.@(js|css|woff2)",
        "headers": [
          {
            "key": "Cache-Control",
            "value": "public, max-age=31536000, immutable"
          }
        ]
      },
      {
        "source": "/",
        "headers": [
          {
            "key": "Cache-Control",
            "value": "no-cache"
          }
        ]
      }
    ]
  },
  "functions": {
    "source": "functions",
    "runtime": "nodejs20",
    "predeploy": ["npm --prefix functions run build"]
  }
}
```

---

## 8. マルチクラウド・ハイブリッドデプロイ

### 8.1 フロントエンド + バックエンド分離パターン

```yaml
# .github/workflows/deploy-multi.yml
name: Multi-Cloud Deploy

on:
  push:
    branches: [main]

jobs:
  # フロントエンドは Vercel にデプロイ
  deploy-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci && npm run build:frontend

      - name: Deploy Frontend to Vercel
        run: |
          npx vercel pull --yes --environment=production --token=${{ secrets.VERCEL_TOKEN }}
          npx vercel build --prod --token=${{ secrets.VERCEL_TOKEN }}
          npx vercel deploy --prebuilt --prod --token=${{ secrets.VERCEL_TOKEN }}

  # バックエンドは AWS ECS にデプロイ
  deploy-backend:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ap-northeast-1

      - name: Login to ECR
        id: ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and Push
        run: |
          docker build -f backend/Dockerfile \
            -t ${{ steps.ecr.outputs.registry }}/api:${{ github.sha }} \
            .
          docker push ${{ steps.ecr.outputs.registry }}/api:${{ github.sha }}

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster prod-cluster \
            --service api-service \
            --force-new-deployment

  # エッジ処理は Cloudflare Workers にデプロイ
  deploy-edge:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: cd edge && npm ci

      - name: Deploy Edge Functions
        uses: cloudflare/wrangler-action@v3
        with:
          apiToken: ${{ secrets.CF_API_TOKEN }}
          accountId: ${{ secrets.CF_ACCOUNT_ID }}
          workingDirectory: edge
          command: deploy

  # 全デプロイ完了後の統合テスト
  integration-test:
    needs: [deploy-frontend, deploy-backend, deploy-edge]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci

      - name: Run E2E Tests
        run: |
          npx playwright test --project=production
        env:
          FRONTEND_URL: https://app.example.com
          API_URL: https://api.example.com
          EDGE_URL: https://edge.example.com
```

### 8.2 DNS ベースのフェイルオーバー

```hcl
# terraform/dns-failover.tf
# Route 53 ヘルスチェック
resource "aws_route53_health_check" "primary" {
  fqdn              = "api-primary.example.com"
  port               = 443
  type               = "HTTPS"
  resource_path      = "/health"
  failure_threshold  = 3
  request_interval   = 10

  tags = {
    Name = "primary-api-health-check"
  }
}

resource "aws_route53_health_check" "secondary" {
  fqdn              = "api-secondary.example.com"
  port               = 443
  type               = "HTTPS"
  resource_path      = "/health"
  failure_threshold  = 3
  request_interval   = 10

  tags = {
    Name = "secondary-api-health-check"
  }
}

# フェイルオーバーレコード
resource "aws_route53_record" "primary" {
  zone_id = var.hosted_zone_id
  name    = "api.example.com"
  type    = "A"

  alias {
    name                   = var.primary_alb_dns
    zone_id                = var.primary_alb_zone_id
    evaluate_target_health = true
  }

  failover_routing_policy {
    type = "PRIMARY"
  }

  health_check_id = aws_route53_health_check.primary.id
  set_identifier  = "primary"
}

resource "aws_route53_record" "secondary" {
  zone_id = var.hosted_zone_id
  name    = "api.example.com"
  type    = "A"

  alias {
    name                   = var.secondary_alb_dns
    zone_id                = var.secondary_alb_zone_id
    evaluate_target_health = true
  }

  failover_routing_policy {
    type = "SECONDARY"
  }

  health_check_id = aws_route53_health_check.secondary.id
  set_identifier  = "secondary"
}
```

---

## 9. 環境管理とシークレット

### 9.1 GitHub Environments によるデプロイ保護

```yaml
# .github/workflows/deploy-with-environments.yml
name: Deploy with Environment Protection

on:
  push:
    branches: [main]

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://staging.example.com
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Staging
        run: echo "Deploying to staging..."

  # 手動承認ゲート（environment protection rules で設定）
  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://www.example.com
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Production
        run: echo "Deploying to production..."
```

### 9.2 シークレット管理のベストプラクティス

```yaml
# AWS Secrets Manager からシークレットを取得
- name: Get Secrets from AWS Secrets Manager
  uses: aws-actions/aws-secretsmanager-get-secrets@v2
  with:
    secret-ids: |
      prod/database
      prod/api-keys
    parse-json-secrets: true

# 使用例
- name: Use Secrets
  run: |
    echo "Database host: $PROD_DATABASE_HOST"  # 自動的に環境変数に展開
  env:
    # Secrets Manager のシークレットが環境変数として利用可能
    DB_URL: ${{ env.PROD_DATABASE_URL }}
```

```typescript
// シークレット管理ユーティリティ（TypeScript）
import {
  SecretsManagerClient,
  GetSecretValueCommand,
} from '@aws-sdk/client-secrets-manager';

const client = new SecretsManagerClient({ region: 'ap-northeast-1' });

interface AppSecrets {
  DATABASE_URL: string;
  API_KEY: string;
  JWT_SECRET: string;
}

// キャッシュ付きシークレット取得
let cachedSecrets: AppSecrets | null = null;
let cacheExpiry = 0;

export async function getSecrets(): Promise<AppSecrets> {
  const now = Date.now();
  if (cachedSecrets && now < cacheExpiry) {
    return cachedSecrets;
  }

  const command = new GetSecretValueCommand({
    SecretId: 'prod/app-secrets',
    VersionStage: 'AWSCURRENT',
  });

  const response = await client.send(command);
  if (!response.SecretString) {
    throw new Error('Secret value is empty');
  }

  cachedSecrets = JSON.parse(response.SecretString) as AppSecrets;
  cacheExpiry = now + 5 * 60 * 1000; // 5分キャッシュ

  return cachedSecrets;
}
```

---

## 10. 監視とロールバック

### 10.1 デプロイ後の自動監視ワークフロー

```yaml
# .github/workflows/post-deploy-monitor.yml
name: Post-Deploy Monitoring

on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
      url:
        required: true
        type: string
      rollback_ref:
        required: true
        type: string

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Health Check Loop
        id: health
        run: |
          URL="${{ inputs.url }}"
          MAX_RETRIES=10
          RETRY_INTERVAL=30
          SUCCESS_COUNT=0
          REQUIRED_SUCCESSES=5

          for i in $(seq 1 $MAX_RETRIES); do
            STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$URL/health" --max-time 10)
            LATENCY=$(curl -s -o /dev/null -w "%{time_total}" "$URL/health" --max-time 10)

            echo "Check $i/$MAX_RETRIES: HTTP $STATUS, Latency: ${LATENCY}s"

            if [ "$STATUS" = "200" ] && [ "$(echo "$LATENCY < 2.0" | bc)" = "1" ]; then
              SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
              echo "Success count: $SUCCESS_COUNT/$REQUIRED_SUCCESSES"
            else
              SUCCESS_COUNT=0
              echo "Reset success count due to failure"
            fi

            if [ "$SUCCESS_COUNT" -ge "$REQUIRED_SUCCESSES" ]; then
              echo "Health check passed consistently"
              echo "healthy=true" >> "$GITHUB_OUTPUT"
              exit 0
            fi

            sleep $RETRY_INTERVAL
          done

          echo "Health check failed after $MAX_RETRIES attempts"
          echo "healthy=false" >> "$GITHUB_OUTPUT"

      - name: Check Error Rate
        if: steps.health.outputs.healthy == 'true'
        id: error-rate
        run: |
          # CloudWatch からエラー率を取得
          ERROR_RATE=$(aws cloudwatch get-metric-statistics \
            --namespace "AWS/ApplicationELB" \
            --metric-name "HTTPCode_Target_5XX_Count" \
            --dimensions Name=LoadBalancer,Value=${{ secrets.ALB_ARN_SUFFIX }} \
            --start-time $(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%S) \
            --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
            --period 300 \
            --statistics Sum \
            --query 'Datapoints[0].Sum' \
            --output text)

          echo "5xx error count in last 5 minutes: $ERROR_RATE"

          if [ "$ERROR_RATE" != "None" ] && [ "$ERROR_RATE" -gt 10 ]; then
            echo "Error rate too high: $ERROR_RATE"
            echo "acceptable=false" >> "$GITHUB_OUTPUT"
          else
            echo "Error rate acceptable"
            echo "acceptable=true" >> "$GITHUB_OUTPUT"
          fi

      - name: Trigger Rollback
        if: steps.health.outputs.healthy == 'false' || steps.error-rate.outputs.acceptable == 'false'
        uses: actions/github-script@v7
        with:
          script: |
            await github.rest.actions.createWorkflowDispatch({
              owner: context.repo.owner,
              repo: context.repo.repo,
              workflow_id: 'rollback.yml',
              ref: 'main',
              inputs: {
                environment: '${{ inputs.environment }}',
                rollback_ref: '${{ inputs.rollback_ref }}',
                reason: 'Automated rollback: health check or error rate threshold exceeded'
              }
            });

      - name: Notify on Slack
        if: always()
        uses: slackapi/slack-github-action@v2.0.0
        with:
          webhook: ${{ secrets.SLACK_WEBHOOK_URL }}
          webhook-type: incoming-webhook
          payload: |
            {
              "text": "Deploy Monitor: ${{ inputs.environment }}",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Deploy Monitoring Result*\n• Environment: `${{ inputs.environment }}`\n• URL: ${{ inputs.url }}\n• Health: ${{ steps.health.outputs.healthy == 'true' && 'PASS' || 'FAIL' }}\n• Error Rate: ${{ steps.error-rate.outputs.acceptable == 'true' && 'OK' || 'HIGH' }}"
                  }
                }
              ]
            }
```

### 10.2 ロールバックワークフロー

```yaml
# .github/workflows/rollback.yml
name: Rollback Deployment

on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options:
          - staging
          - production
      rollback_ref:
        description: 'Git ref to rollback to (commit SHA or tag)'
        required: true
        type: string
      reason:
        description: 'Reason for rollback'
        required: true
        type: string

jobs:
  rollback:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    permissions:
      id-token: write
      contents: read

    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ inputs.rollback_ref }}

      - name: Log Rollback Initiation
        run: |
          echo "========================================="
          echo "ROLLBACK INITIATED"
          echo "Environment: ${{ inputs.environment }}"
          echo "Rolling back to: ${{ inputs.rollback_ref }}"
          echo "Reason: ${{ inputs.reason }}"
          echo "Initiated by: ${{ github.actor }}"
          echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
          echo "========================================="

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ap-northeast-1

      - name: Rollback ECS Service
        if: inputs.environment == 'production'
        run: |
          # 前回の安定版タスク定義を取得
          PREVIOUS_TD=$(aws ecs describe-services \
            --cluster prod-cluster \
            --services api-service \
            --query 'services[0].deployments[?status==`ACTIVE`].taskDefinition | [0]' \
            --output text)

          if [ "$PREVIOUS_TD" = "None" ]; then
            echo "No previous task definition found. Using image from rollback ref."
            # Git ref から Docker image を再ビルドしてデプロイ
            # ...
          else
            echo "Rolling back to task definition: $PREVIOUS_TD"
            aws ecs update-service \
              --cluster prod-cluster \
              --service api-service \
              --task-definition "$PREVIOUS_TD" \
              --force-new-deployment

            aws ecs wait services-stable \
              --cluster prod-cluster \
              --services api-service
          fi

      - name: Verify Rollback
        run: |
          sleep 30
          STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://api.example.com/health")
          if [ "$STATUS" = "200" ]; then
            echo "Rollback verified successfully"
          else
            echo "WARNING: Rollback verification failed with HTTP $STATUS"
            exit 1
          fi

      - name: Create Rollback Record
        uses: actions/github-script@v7
        with:
          script: |
            await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `[Rollback] ${context.payload.inputs.environment} - ${new Date().toISOString().split('T')[0]}`,
              body: [
                '## Rollback Record',
                '',
                `- **Environment**: ${context.payload.inputs.environment}`,
                `- **Rolled back to**: ${context.payload.inputs.rollback_ref}`,
                `- **Reason**: ${context.payload.inputs.reason}`,
                `- **Initiated by**: ${context.actor}`,
                `- **Timestamp**: ${new Date().toISOString()}`,
              ].join('\n'),
              labels: ['rollback', 'incident']
            });
```

---

## 11. プラットフォーム比較表

| 特性 | AWS (Lambda/ECS) | Vercel | Cloudflare Workers | GCP Cloud Run |
|------|------------------|--------|-------------------|---------------|
| 対象 | バックエンド全般 | フロントエンド+API | エッジAPI | コンテナ全般 |
| コールドスタート | 100ms〜数秒 | 数十ms | ほぼ0ms (V8 Isolate) | 数百ms〜数秒 |
| 最大実行時間 | 15分 (Lambda) | 10秒〜5分 | 30秒 (CPU 50ms) | 60分 |
| メモリ上限 | 10GB (Lambda) | 1024MB | 128MB | 32GB |
| ランタイム | Node.js, Python, Go等 | Node.js | JavaScript/WASM | 任意(Docker) |
| DB 統合 | RDS, DynamoDB, Aurora | Vercel Postgres, KV | D1, KV, Durable Objects | Cloud SQL, Firestore |
| 料金体系 | 従量課金(複雑) | 無料枠+従量 | 無料枠10万req/日 | 従量課金(秒単位) |
| 学習コスト | 高い | 低い | 中 | 中 |
| VPC 接続 | ネイティブ対応 | 非対応 | Tunnel 経由 | VPC コネクタ |
| カスタムドメイン | Route 53 | 自動SSL | 自動SSL | Cloud DNS |
| ロールバック | 手動/自動 | Instant Rollback | Wrangler rollback | リビジョン切替 |

| デプロイ方法比較 | Git 連携 | CLI | IaC (CDK/Terraform) |
|-----------------|---------|-----|---------------------|
| 自動化レベル | 高い | 中 | 最高 |
| 再現性 | 中 | 低い | 最高 |
| 複雑さ | 低い | 低い | 高い |
| 適用場面 | フロント / 小規模API | 開発/テスト | 本番インフラ全般 |
| ロールバック | Git revert | 手動 | 状態管理で自動 |

### コスト比較シミュレーション

```
月間 100万リクエスト、平均レスポンス 50ms の API の場合:

AWS Lambda:
  - リクエスト: $0.20 (100万 × $0.0000002)
  - コンピューティング: $0.83 (128MB, 50ms × 100万)
  - API Gateway: $3.50
  - 合計: 約 $4.53/月

Cloudflare Workers:
  - 無料枠: 10万req/日 = 月300万req → 無料
  - Paid plan ($5/月): 1000万req含む → $5.00/月
  - 合計: $0〜5.00/月

Vercel:
  - Hobby (無料): 100GB帯域まで → $0
  - Pro ($20/月): 1TB帯域、Serverless 1000時間 → $20/月
  - 合計: $0〜20.00/月

GCP Cloud Run:
  - CPU: $0.00002400/vCPU秒 × 50,000秒 = $1.20
  - メモリ: $0.00000250/GiB秒 × 50,000秒 × 0.5GiB = $0.06
  - リクエスト: $0.40 (100万 × $0.0000004)
  - 合計: 約 $1.66/月（最小インスタンス0の場合）
```

---

## 12. アンチパターン

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

### アンチパターン 3: デプロイ後のヘルスチェック不在

```yaml
# 悪い例: デプロイして終わり
- name: Deploy
  run: npx vercel deploy --prod
# → デプロイ後にアプリが正常に動作しているか確認しない

# 良い例: デプロイ後にヘルスチェックを実行
- name: Deploy
  id: deploy
  run: |
    URL=$(npx vercel deploy --prod)
    echo "url=${URL}" >> "$GITHUB_OUTPUT"

- name: Health Check
  run: |
    URL="${{ steps.deploy.outputs.url }}"
    for i in $(seq 1 5); do
      STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${URL}/api/health" --max-time 10)
      if [ "$STATUS" = "200" ]; then
        echo "Health check passed on attempt $i"
        exit 0
      fi
      echo "Attempt $i failed (HTTP $STATUS), retrying in 10s..."
      sleep 10
    done
    echo "Health check failed after 5 attempts"
    exit 1
```

### アンチパターン 4: 単一リージョン依存

```
[悪い例]
- 全リソースを ap-northeast-1（東京）のみに配置
- 東京リージョン障害時にサービス全停止

[良い例]
- CloudFront / Cloudflare で静的コンテンツをグローバルに配信
- クリティカルな API は複数リージョンに配置
- Route 53 ヘルスチェックによるフェイルオーバー設定
- データベースは Aurora Global Database またはリードレプリカ
```

### アンチパターン 5: ビルド成果物の非再現性

```yaml
# 悪い例: ビルドのたびに異なる結果になる可能性
- run: |
    npm install  # package-lock.json を無視
    npm run build

# 良い例: 再現可能なビルド
- run: |
    npm ci                     # lock ファイルに厳密に従う
    npm run build
  env:
    NODE_ENV: production
    NEXT_TELEMETRY_DISABLED: 1  # テレメトリ無効化で確定的なビルド
```

---

## 13. FAQ

### Q1: Vercel と AWS、どちらを選ぶべきですか？

フロントエンド（Next.js/React）が中心で、バックエンドが軽量な API Routes 程度なら Vercel が圧倒的に楽です。複雑なバックエンド処理、VPC 内のリソースアクセス、長時間バッチ処理が必要な場合は AWS を選択してください。多くのプロジェクトでは「フロントは Vercel、バックエンドは AWS」という組み合わせが実用的です。

### Q2: Cloudflare Workers の CPU 制限 (50ms) は厳しすぎませんか？

CPU 時間 50ms は「I/O 待ち時間を除いた純粋な計算時間」です。データベースクエリや外部 API 呼び出しの待ち時間は含まれません。一般的な API 処理（JSON パース、バリデーション、レスポンス構築）は数 ms で完了するため、ほとんどのユースケースでは十分です。重い計算処理が必要な場合は Workers Unbound（CPU 時間 30 秒）を検討してください。

### Q3: OIDC による AWS 認証とは何ですか？ なぜ推奨されるのですか？

GitHub Actions から AWS にアクセスする際、従来は IAM ユーザーのアクセスキーをシークレットに保存していました。OIDC（OpenID Connect）では、GitHub が発行する短命トークンを AWS が直接検証するため、長期間有効なシークレットの管理が不要になります。キーローテーションの手間がなく、漏洩リスクも低減されます。

### Q4: Cloud Run と Lambda、どちらを選ぶべきですか？

**Lambda が向いているケース:**
- イベント駆動型処理（S3 アップロード、SQS メッセージ）
- 既存の AWS サービスとの統合が多い
- コールドスタートが許容される軽量 API
- 従量課金でコストを最小化したい

**Cloud Run が向いているケース:**
- 既存の Docker コンテナをそのままデプロイしたい
- リクエスト処理時間が長い（15分超）
- WebSocket やストリーミングが必要
- ポータブルなコンテナ環境を維持したい

### Q5: Vercel の Preview Deployment を効率的に使うコツは？

```
1. ブランチごとにプレビューURLが自動生成される
   → PR レビューで実際の動作を確認可能

2. 環境変数のスコープを適切に設定
   → Preview / Production で異なるDB接続先を使用

3. Vercel CLI でローカルからプレビューデプロイ
   → CI を待たずに確認: npx vercel deploy

4. PR コメントにプレビューURLを自動投稿（GitHub Integration）
   → レビュアーがワンクリックで確認可能

5. プレビュー環境でのみ有効な機能フラグ
   → 未完成機能をプレビューで確認、本番には影響なし
```

### Q6: マルチクラウド構成のデメリットは何ですか？

マルチクラウドには可用性向上やベンダーロックイン回避のメリットがある一方、以下のデメリットがあります:

- **運用複雑性の増大**: 各クラウドの IAM、ネットワーク、監視を個別に管理する必要がある
- **学習コスト**: チーム全員が複数のクラウドに精通する必要がある
- **データ転送コスト**: クラウド間のデータ転送には Egress 料金が発生する
- **整合性の確保**: 複数のクラウドにまたがるトランザクションの管理が困難
- **ツール統一の困難**: Terraform で抽象化しても、各プロバイダ固有の設定は残る

推奨アプローチ: 明確な理由がない限り、メインクラウドを1つ選び、エッジ処理（Cloudflare）やフロントエンド（Vercel）のみ別プラットフォームを使う「ハイブリッド」が現実的です。

### Q7: ECS Fargate と EKS（Kubernetes）の使い分けは？

```
ECS Fargate が適している:
  - AWS ネイティブなサービス連携（ALB, CloudWatch, Secrets Manager）
  - 小〜中規模のマイクロサービス（10サービス程度まで）
  - Kubernetes の学習コストを避けたい
  - AWS 以外のクラウドへの移植性が不要

EKS が適している:
  - 大規模なマイクロサービス（数十〜数百サービス）
  - Kubernetes エコシステム（Istio, Argo, Helm）を活用したい
  - マルチクラウド/ハイブリッドクラウドでの一貫した運用
  - 高度なトラフィック制御（Service Mesh）が必要
  - チームに Kubernetes の知見がある
```

---

## まとめ

| 項目 | 要点 |
|------|------|
| AWS S3+CloudFront | 静的サイトの定番。IaC で管理し、CloudFront で高速配信。OAC による安全なアクセス制御 |
| AWS Lambda (SAM) | サーバーレス API。Canary デプロイとアラーム連携が容易。Layer で依存関係を分離 |
| AWS ECS/Fargate | コンテナデプロイの標準。Circuit Breaker で自動ロールバック。Auto Scaling で負荷対応 |
| Vercel | Next.js 最適化。プレビュー環境と Git 連携が強力。Edge Functions でエッジ処理 |
| Cloudflare Workers | エッジ実行でレイテンシ最小。D1/KV/R2/Durable Objects のエコシステム |
| GCP Cloud Run | Docker コンテナをそのままデプロイ。段階的トラフィック移行が容易 |
| OIDC 認証 | CI/CD からのクラウド認証はシークレットキーより OIDC を推奨 |
| キャッシュ戦略 | アセットは immutable、HTML は no-cache、API は stale-while-revalidate |
| マルチクラウド | フロント(Vercel) + バックエンド(AWS) + エッジ(Cloudflare) のハイブリッドが現実的 |
| 監視とロールバック | デプロイ後のヘルスチェック必須。自動ロールバック機構を組み込む |

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
5. **AWS ECS Developer Guide** — https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ — ECS のベストプラクティス
6. **Google Cloud Run Documentation** — https://cloud.google.com/run/docs — Cloud Run の公式ドキュメント
7. **Firebase Hosting Documentation** — https://firebase.google.com/docs/hosting — Firebase Hosting のガイド
8. **Terraform AWS Provider** — https://registry.terraform.io/providers/hashicorp/aws/ — Terraform による AWS リソース管理

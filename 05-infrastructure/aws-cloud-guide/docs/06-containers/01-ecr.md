# Amazon ECR (Elastic Container Registry)

> コンテナイメージの保存・管理を行う Amazon ECR のリポジトリ作成、イメージのプッシュ/プル、ライフサイクルポリシー、イメージスキャンまでを体系的に学ぶ。

---

## この章で学ぶこと

1. **ECR リポジトリの作成と管理** -- プライベート/パブリックリポジトリの作成、アクセス制御の基本を理解する
2. **イメージのビルド・プッシュ・プル** -- Docker CLI を使ったイメージ操作と ECR 認証の仕組みを習得する
3. **ライフサイクルポリシーとイメージスキャン** -- 不要イメージの自動削除と脆弱性スキャンでセキュリティと運用コストを管理する
4. **クロスリージョン/クロスアカウントレプリケーション** -- マルチリージョンやマルチアカウント環境でのイメージ配布戦略を学ぶ
5. **CI/CD パイプラインとの統合** -- ECR をビルドパイプラインに組み込む実践的な手法を習得する
6. **セキュリティベストプラクティス** -- イメージ署名、非rootユーザー実行、脆弱性対応の自動化を実装する


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [Amazon ECS 基礎](./00-ecs-basics.md) の内容を理解していること

---

## 1. ECR の概要

### 1.1 ECR のアーキテクチャ

```
開発者 / CI/CD
    |
    | docker push / pull
    v
+--------------------------------+
| Amazon ECR                     |
| +----------------------------+ |
| | プライベートリポジトリ       | |
| | +--------+ +--------+     | |
| | | my-app | | my-api |     | |
| | | :v1.0  | | :v2.1  |     | |
| | | :v1.1  | | :latest|     | |
| | +--------+ +--------+     | |
| +----------------------------+ |
| +----------------------------+ |
| | パブリックリポジトリ         | |
| | (ECR Public Gallery)       | |
| +----------------------------+ |
+--------------------------------+
    |               |
    v               v
+----------+  +----------+
| ECS      |  | EKS      |
| Fargate  |  | ノード    |
+----------+  +----------+
```

### 1.2 ECR の特徴

| 特徴 | 説明 |
|------|------|
| フルマネージド | インフラ管理不要、高可用性 |
| 暗号化 | 保存時暗号化 (AES-256 or KMS) |
| IAM 統合 | きめ細かなアクセス制御 |
| イメージスキャン | 脆弱性の自動検出 |
| レプリケーション | クロスリージョン/クロスアカウント |
| イメージ署名 | コンテンツの信頼性検証 |
| ライフサイクルポリシー | 不要イメージの自動削除 |
| OCI 互換 | OCI アーティファクト(Helm チャート等)の保存に対応 |

### 1.3 ECR の料金体系

```
ECR の料金構成:

1. ストレージ料金:
   $0.10/GB/月 (プライベートリポジトリ)

2. データ転送:
   - 同一リージョン内: 無料
   - リージョン間: 標準データ転送料金
   - インターネットへ: 標準データ転送料金

3. プルスルーキャッシュ:
   - Docker Hub 等からの初回プル: データ転送料金
   - キャッシュからのプル: 無料

コスト試算例:
  イメージサイズ: 500 MB x 20 バージョン = 10 GB
  月額: 10 GB x $0.10 = $1.00/月

  ライフサイクルポリシーで保持数を制限することで
  ストレージコストを大幅に削減可能
```

---

## 2. リポジトリの作成

### 2.1 AWS CLI によるリポジトリ作成

```bash
# プライベートリポジトリの作成
aws ecr create-repository \
  --repository-name my-app \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration encryptionType=AES256 \
  --image-tag-mutability IMMUTABLE

# 出力例:
# {
#   "repository": {
#     "repositoryArn": "arn:aws:ecr:ap-northeast-1:123456789012:repository/my-app",
#     "repositoryUri": "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-app",
#     "repositoryName": "my-app"
#   }
# }

# KMS 暗号化を使用するリポジトリの作成
aws ecr create-repository \
  --repository-name my-secure-app \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration '{
    "encryptionType": "KMS",
    "kmsKey": "arn:aws:kms:ap-northeast-1:123456789012:key/12345678-1234-1234-1234-123456789012"
  }' \
  --image-tag-mutability IMMUTABLE

# 名前空間付きリポジトリの作成 (組織構造を反映)
aws ecr create-repository --repository-name team-a/frontend
aws ecr create-repository --repository-name team-a/backend
aws ecr create-repository --repository-name team-b/data-pipeline
aws ecr create-repository --repository-name shared/base-images

# リポジトリの一覧表示
aws ecr describe-repositories \
  --query 'repositories[*].{Name: repositoryName, URI: repositoryUri, Scanning: imageScanningConfiguration.scanOnPush, TagMutability: imageTagMutability}'

# リポジトリの削除 (イメージが含まれている場合は --force が必要)
aws ecr delete-repository \
  --repository-name my-old-app \
  --force
```

### 2.2 タグの不変性 (Immutability)

```
タグの不変性設定:

MUTABLE (デフォルト):
  push my-app:v1.0  -->  イメージA を v1.0 で保存
  push my-app:v1.0  -->  イメージB で v1.0 を上書き ← 危険！

IMMUTABLE (推奨):
  push my-app:v1.0  -->  イメージA を v1.0 で保存
  push my-app:v1.0  -->  エラー！既存タグは上書き不可
  push my-app:v1.1  -->  イメージB を v1.1 で保存 ← OK
```

| 設定 | MUTABLE | IMMUTABLE |
|------|---------|-----------|
| 同一タグの上書き | 可能 | 不可 |
| デプロイの再現性 | 低い | 高い |
| 監査追跡 | 困難 | 容易 |
| 推奨環境 | 開発 | 本番 |

```bash
# タグの不変性設定の変更
aws ecr put-image-tag-mutability \
  --repository-name my-app \
  --image-tag-mutability IMMUTABLE
```

### 2.3 リポジトリポリシー (クロスアカウントアクセス)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowPullFromProdAccount",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::987654321098:root"
      },
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability"
      ]
    },
    {
      "Sid": "AllowPullFromSpecificRole",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::987654321098:role/ECSTaskExecutionRole"
      },
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability"
      ]
    },
    {
      "Sid": "AllowPushFromCICD",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::111222333444:role/CICDPipelineRole"
      },
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ]
    }
  ]
}
```

```bash
# リポジトリポリシーの適用
aws ecr set-repository-policy \
  --repository-name my-app \
  --policy-text file://ecr-policy.json

# リポジトリポリシーの確認
aws ecr get-repository-policy \
  --repository-name my-app

# リポジトリポリシーの削除
aws ecr delete-repository-policy \
  --repository-name my-app
```

### 2.4 レジストリレベルのポリシー

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowPullFromOrganization",
      "Effect": "Allow",
      "Principal": "*",
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability"
      ],
      "Condition": {
        "StringEquals": {
          "aws:PrincipalOrgID": "o-1234567890"
        }
      }
    }
  ]
}
```

```bash
# レジストリポリシーの設定
aws ecr put-registry-policy \
  --policy-text file://registry-policy.json

# レジストリポリシーの確認
aws ecr get-registry-policy
```

---

## 3. イメージのビルドとプッシュ

### 3.1 ECR 認証とプッシュの流れ

```
イメージプッシュのフロー:

1. ECR 認証トークン取得
   aws ecr get-login-password
        |
        v
2. Docker ログイン
   docker login --username AWS --password <token>
        |
        v
3. イメージビルド
   docker build -t my-app:v1.0 .
        |
        v
4. タグ付け
   docker tag my-app:v1.0 <ECR_URI>/my-app:v1.0
        |
        v
5. プッシュ
   docker push <ECR_URI>/my-app:v1.0
        |
        v
6. ECR に保存 (暗号化、スキャン)
```

### 3.2 実際のコマンド

```bash
# 変数定義
AWS_ACCOUNT_ID=123456789012
REGION=ap-northeast-1
REPO_NAME=my-app
IMAGE_TAG=v1.0.0
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# 1. ECR にログイン
aws ecr get-login-password --region ${REGION} | \
  docker login --username AWS --password-stdin ${ECR_URI}

# 2. イメージのビルド
docker build -t ${REPO_NAME}:${IMAGE_TAG} .

# 3. ECR 用にタグ付け
docker tag ${REPO_NAME}:${IMAGE_TAG} ${ECR_URI}/${REPO_NAME}:${IMAGE_TAG}

# 4. プッシュ
docker push ${ECR_URI}/${REPO_NAME}:${IMAGE_TAG}

# 5. イメージ一覧の確認
aws ecr list-images --repository-name ${REPO_NAME}

# 6. イメージの詳細確認
aws ecr describe-images \
  --repository-name ${REPO_NAME} \
  --image-ids imageTag=${IMAGE_TAG} \
  --query 'imageDetails[0].{
    Digest: imageDigest,
    Tags: imageTags,
    Size: imageSizeInBytes,
    PushedAt: imagePushedAt,
    ScanStatus: imageScanStatus.status,
    ScanFindings: imageScanFindingsSummary
  }'

# 7. イメージの削除
aws ecr batch-delete-image \
  --repository-name ${REPO_NAME} \
  --image-ids imageTag=v0.9.0

# 8. マルチアーキテクチャビルド (amd64 + arm64)
docker buildx create --use
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ${ECR_URI}/${REPO_NAME}:${IMAGE_TAG} \
  --push .
```

### 3.3 マルチステージビルドの Dockerfile 例

```dockerfile
# ---- ビルドステージ ----
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# ---- 本番ステージ ----
FROM node:20-alpine AS production
WORKDIR /app

# セキュリティ: 非rootユーザーで実行
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup
USER appuser

COPY --from=builder --chown=appuser:appgroup /app/dist ./dist
COPY --from=builder --chown=appuser:appgroup /app/node_modules ./node_modules
COPY --from=builder --chown=appuser:appgroup /app/package.json ./

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -qO- http://localhost:8080/health || exit 1

CMD ["node", "dist/server.js"]
```

### 3.4 Go アプリケーションの軽量イメージ

```dockerfile
# Go アプリケーション用の超軽量イメージ
FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s" -o /app/server ./cmd/server

# distroless イメージで実行 (シェルすらない超軽量イメージ)
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /app/server /server

USER nonroot:nonroot
EXPOSE 8080

ENTRYPOINT ["/server"]
```

### 3.5 Python アプリケーションの最適化

```dockerfile
# Python アプリケーション用の最適化 Dockerfile
FROM python:3.12-slim AS builder

WORKDIR /app

# 仮想環境を使って依存関係を分離
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 本番ステージ
FROM python:3.12-slim

# セキュリティパッチの適用
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 非rootユーザーの作成
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

WORKDIR /app

# ビルドステージから仮想環境をコピー
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=appuser:appuser . .

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "app:app"]
```

### 3.6 .dockerignore の設定

```
# .dockerignore
.git
.gitignore
.env
.env.local
.env.*.local
*.md
!README.md
docker-compose*.yml
Dockerfile*
.dockerignore
node_modules
__pycache__
*.pyc
.pytest_cache
.coverage
coverage/
dist/
build/
.vscode
.idea
*.log
tmp/
temp/
tests/
test/
docs/
```

---

## 4. ライフサイクルポリシー

### 4.1 ポリシーの仕組み

```
ライフサイクルポリシーの動作:

リポジトリ内イメージ:
  v1.0  (30日前)  ←── 古いイメージを自動削除
  v1.1  (20日前)  ←── 古いイメージを自動削除
  v1.2  (10日前)
  v1.3  (5日前)
  v1.4  (2日前)
  v1.5  (今日)     ←── 最新N個は保持

ポリシー適用後:
  v1.2  (10日前)   ←── 保持 (最新4個)
  v1.3  (5日前)    ←── 保持
  v1.4  (2日前)    ←── 保持
  v1.5  (今日)     ←── 保持

ポリシーの評価順序:
  1. rulePriority の低い順に評価
  2. 各ルールのフィルタに一致するイメージを選択
  3. 条件に基づいてイメージを期限切れにマーク
  4. 低い優先度で一致したイメージは高い優先度のルールでは評価されない
```

### 4.2 ライフサイクルポリシーの設定

```json
{
  "rules": [
    {
      "rulePriority": 1,
      "description": "untagged イメージを7日後に削除",
      "selection": {
        "tagStatus": "untagged",
        "countType": "sinceImagePushed",
        "countUnit": "days",
        "countNumber": 7
      },
      "action": {
        "type": "expire"
      }
    },
    {
      "rulePriority": 2,
      "description": "dev タグは最新5個を保持",
      "selection": {
        "tagStatus": "tagged",
        "tagPrefixList": ["dev-"],
        "countType": "imageCountMoreThan",
        "countNumber": 5
      },
      "action": {
        "type": "expire"
      }
    },
    {
      "rulePriority": 3,
      "description": "stg タグは最新10個を保持",
      "selection": {
        "tagStatus": "tagged",
        "tagPrefixList": ["stg-"],
        "countType": "imageCountMoreThan",
        "countNumber": 10
      },
      "action": {
        "type": "expire"
      }
    },
    {
      "rulePriority": 4,
      "description": "release タグは最新100個を保持",
      "selection": {
        "tagStatus": "tagged",
        "tagPrefixList": ["v", "release-"],
        "countType": "imageCountMoreThan",
        "countNumber": 100
      },
      "action": {
        "type": "expire"
      }
    },
    {
      "rulePriority": 5,
      "description": "その他のタグ付きイメージは90日で削除",
      "selection": {
        "tagStatus": "any",
        "countType": "sinceImagePushed",
        "countUnit": "days",
        "countNumber": 90
      },
      "action": {
        "type": "expire"
      }
    }
  ]
}
```

```bash
# ライフサイクルポリシーの適用
aws ecr put-lifecycle-policy \
  --repository-name my-app \
  --lifecycle-policy-text file://lifecycle-policy.json

# ポリシーの確認
aws ecr get-lifecycle-policy \
  --repository-name my-app

# ポリシーのプレビュー (ドライラン)
aws ecr start-lifecycle-policy-preview \
  --repository-name my-app \
  --lifecycle-policy-text file://lifecycle-policy.json

# プレビュー結果の確認
aws ecr get-lifecycle-policy-preview \
  --repository-name my-app

# ポリシーの削除
aws ecr delete-lifecycle-policy \
  --repository-name my-app

# 全リポジトリに一括適用するスクリプト
REPOS=$(aws ecr describe-repositories --query 'repositories[*].repositoryName' --output text)
for REPO in $REPOS; do
  echo "Applying lifecycle policy to ${REPO}..."
  aws ecr put-lifecycle-policy \
    --repository-name "${REPO}" \
    --lifecycle-policy-text file://lifecycle-policy.json
done
```

---

## 5. イメージスキャン

### 5.1 スキャンの種類

```
スキャン方式の比較:

ベーシックスキャン (無料):
  プッシュ時 --> Clair エンジン --> OS パッケージの CVE 検出
  手動実行可能

拡張スキャン (Amazon Inspector 統合):
  プッシュ時 --> Inspector --> OS パッケージ + プログラミング言語
  継続的スキャン                パッケージの CVE 検出
  (新しいCVE発見時に自動再スキャン)
```

| 機能 | ベーシックスキャン | 拡張スキャン |
|------|-----------------|-------------|
| 料金 | 無料 | Amazon Inspector 料金 |
| スキャン対象 | OS パッケージ | OS + 言語パッケージ |
| トリガー | プッシュ時/手動 | プッシュ時 + 継続的 |
| 新規 CVE 対応 | 手動再スキャン | 自動再スキャン |
| EventBridge 連携 | あり | あり |
| 対応言語 | - | Java, Python, Node.js, Go, Ruby, .NET |
| SBOM 生成 | なし | あり |

### 5.2 スキャン結果の確認

```bash
# ベーシックスキャンの有効化 (リポジトリ単位)
aws ecr put-image-scanning-configuration \
  --repository-name my-app \
  --image-scanning-configuration scanOnPush=true

# 拡張スキャンの有効化 (レジストリ単位)
aws ecr put-registry-scanning-configuration \
  --scan-type ENHANCED \
  --rules '[
    {
      "repositoryFilters": [
        {
          "filter": "*",
          "filterType": "WILDCARD"
        }
      ],
      "scanFrequency": "CONTINUOUS_SCAN"
    }
  ]'

# 手動スキャンの開始
aws ecr start-image-scan \
  --repository-name my-app \
  --image-id imageTag=v1.0.0

# スキャン結果の確認
aws ecr describe-image-scan-findings \
  --repository-name my-app \
  --image-id imageTag=v1.0.0

# 重要度別のフィルタリング
aws ecr describe-image-scan-findings \
  --repository-name my-app \
  --image-id imageTag=v1.0.0 \
  --query 'imageScanFindings.findingsSeverityCounts'

# CRITICAL と HIGH の脆弱性のみ表示
aws ecr describe-image-scan-findings \
  --repository-name my-app \
  --image-id imageTag=v1.0.0 \
  --query 'imageScanFindings.findings[?severity==`CRITICAL` || severity==`HIGH`].{
    Name: name,
    Severity: severity,
    Description: description,
    URI: uri
  }'
```

### 5.3 スキャン結果に基づく自動通知

```python
# EventBridge ルールで ECR スキャン完了イベントを検知し、
# CRITICAL/HIGH 脆弱性があれば Slack に通知する Lambda
import json
import os
import urllib3

SLACK_WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]

def lambda_handler(event, context):
    detail = event["detail"]
    repo = detail["repository-name"]
    tag = detail["image-tags"][0] if detail.get("image-tags") else "untagged"
    severity_counts = detail["finding-severity-counts"]

    critical = severity_counts.get("CRITICAL", 0)
    high = severity_counts.get("HIGH", 0)
    medium = severity_counts.get("MEDIUM", 0)

    if critical > 0 or high > 0:
        color = "#ff0000" if critical > 0 else "#ff9900"
        message = {
            "attachments": [
                {
                    "color": color,
                    "title": f"ECR Image Scan Alert: {repo}:{tag}",
                    "fields": [
                        {"title": "CRITICAL", "value": str(critical), "short": True},
                        {"title": "HIGH", "value": str(high), "short": True},
                        {"title": "MEDIUM", "value": str(medium), "short": True},
                        {"title": "Repository", "value": repo, "short": True}
                    ],
                    "text": "脆弱性が検出されました。確認してください。",
                    "footer": "Amazon ECR Image Scan"
                }
            ]
        }
        http = urllib3.PoolManager()
        http.request("POST", SLACK_WEBHOOK_URL,
                     body=json.dumps(message),
                     headers={"Content-Type": "application/json"})

    return {"statusCode": 200, "body": "Processed"}
```

```yaml
# EventBridge ルール (CloudFormation)
ECRScanEventRule:
  Type: AWS::Events::Rule
  Properties:
    Name: ecr-scan-findings
    Description: ECR スキャン完了イベントを検知
    EventPattern:
      source:
        - "aws.ecr"
      detail-type:
        - "ECR Image Scan"
      detail:
        scan-status:
          - "COMPLETE"
        finding-severity-counts:
          CRITICAL:
            - numeric: [">", 0]
    Targets:
      - Arn: !GetAtt ScanNotificationFunction.Arn
        Id: ScanNotification
```

### 5.4 CI/CD パイプラインでのスキャンゲート

```bash
#!/bin/bash
# CI/CD パイプラインでのイメージスキャンゲート
# CRITICAL/HIGH 脆弱性がある場合はデプロイをブロック

REPO_NAME=$1
IMAGE_TAG=$2
MAX_CRITICAL=0
MAX_HIGH=0

echo "Waiting for scan to complete..."
aws ecr wait image-scan-complete \
  --repository-name ${REPO_NAME} \
  --image-id imageTag=${IMAGE_TAG}

# スキャン結果の取得
FINDINGS=$(aws ecr describe-image-scan-findings \
  --repository-name ${REPO_NAME} \
  --image-id imageTag=${IMAGE_TAG} \
  --query 'imageScanFindings.findingsSeverityCounts')

CRITICAL=$(echo ${FINDINGS} | jq -r '.CRITICAL // 0')
HIGH=$(echo ${FINDINGS} | jq -r '.HIGH // 0')

echo "Scan Results: CRITICAL=${CRITICAL}, HIGH=${HIGH}"

if [ "${CRITICAL}" -gt "${MAX_CRITICAL}" ]; then
  echo "ERROR: ${CRITICAL} CRITICAL vulnerabilities found. Maximum allowed: ${MAX_CRITICAL}"
  exit 1
fi

if [ "${HIGH}" -gt "${MAX_HIGH}" ]; then
  echo "ERROR: ${HIGH} HIGH vulnerabilities found. Maximum allowed: ${MAX_HIGH}"
  exit 1
fi

echo "Scan passed. Proceeding with deployment."
exit 0
```

---

## 6. クロスリージョン/クロスアカウントレプリケーション

### 6.1 レプリケーション設定

```
レプリケーション構成:

ソースリージョン (ap-northeast-1)
+------------------+
| ECR: my-app      |  push
| :v1.0            | ----+
+------------------+     |
                         | 自動レプリケーション
                         |
    +--------------------+--------------------+
    |                                         |
    v                                         v
宛先リージョン (us-east-1)           宛先アカウント (987654321098)
+------------------+               +------------------+
| ECR: my-app      |               | ECR: my-app      |
| :v1.0 (複製)     |               | :v1.0 (複製)     |
+------------------+               +------------------+
```

```bash
# クロスリージョンレプリケーション設定
aws ecr put-replication-configuration \
  --replication-configuration '{
    "rules": [
      {
        "destinations": [
          {
            "region": "us-east-1",
            "registryId": "123456789012"
          },
          {
            "region": "eu-west-1",
            "registryId": "123456789012"
          }
        ],
        "repositoryFilters": [
          {
            "filter": "prod-",
            "filterType": "PREFIX_MATCH"
          }
        ]
      }
    ]
  }'

# クロスアカウントレプリケーション設定
aws ecr put-replication-configuration \
  --replication-configuration '{
    "rules": [
      {
        "destinations": [
          {
            "region": "ap-northeast-1",
            "registryId": "987654321098"
          }
        ],
        "repositoryFilters": [
          {
            "filter": "shared/",
            "filterType": "PREFIX_MATCH"
          }
        ]
      }
    ]
  }'

# レプリケーション設定の確認
aws ecr describe-registry \
  --query 'replicationConfiguration'

# 宛先アカウント側でのレジストリポリシー設定
# (レプリケーションを受け入れるために必要)
aws ecr put-registry-policy \
  --policy-text '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Sid": "AllowReplication",
        "Effect": "Allow",
        "Principal": {
          "AWS": "arn:aws:iam::123456789012:root"
        },
        "Action": [
          "ecr:CreateRepository",
          "ecr:ReplicateImage"
        ],
        "Resource": "arn:aws:ecr:ap-northeast-1:987654321098:repository/*"
      }
    ]
  }'
```

### 6.2 プルスルーキャッシュ

```
プルスルーキャッシュの仕組み:

1回目のプル:
  ECS/EKS --> ECR (キャッシュなし) --> Docker Hub --> イメージ取得
                    |
                    v
              キャッシュに保存

2回目以降のプル:
  ECS/EKS --> ECR (キャッシュあり) --> イメージ取得 (高速)
  Docker Hub への通信なし = レート制限の影響を回避
```

```bash
# プルスルーキャッシュルールの作成
aws ecr create-pull-through-cache-rule \
  --ecr-repository-prefix docker-hub \
  --upstream-registry-url registry-1.docker.io

# GitHub Container Registry のキャッシュ
aws ecr create-pull-through-cache-rule \
  --ecr-repository-prefix ghcr \
  --upstream-registry-url ghcr.io

# Quay.io のキャッシュ
aws ecr create-pull-through-cache-rule \
  --ecr-repository-prefix quay \
  --upstream-registry-url quay.io

# プルスルーキャッシュの利用
# docker pull 123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/docker-hub/library/nginx:latest
# → 初回は Docker Hub からプル、以降は ECR キャッシュからプル

# キャッシュルールの一覧
aws ecr describe-pull-through-cache-rules

# キャッシュルールの削除
aws ecr delete-pull-through-cache-rule \
  --ecr-repository-prefix docker-hub
```

---

## 7. CI/CD パイプラインとの統合

### 7.1 GitHub Actions での ECR 統合

```yaml
# .github/workflows/build-and-push.yml
name: Build and Push to ECR

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: ap-northeast-1
  ECR_REPOSITORY: my-app

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsRole
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, tag, and push image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:latest .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

      - name: Wait for scan and check results
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          aws ecr wait image-scan-complete \
            --repository-name $ECR_REPOSITORY \
            --image-id imageTag=$IMAGE_TAG
          CRITICAL=$(aws ecr describe-image-scan-findings \
            --repository-name $ECR_REPOSITORY \
            --image-id imageTag=$IMAGE_TAG \
            --query 'imageScanFindings.findingsSeverityCounts.CRITICAL' \
            --output text)
          if [ "$CRITICAL" != "None" ] && [ "$CRITICAL" -gt 0 ]; then
            echo "CRITICAL vulnerabilities found: $CRITICAL"
            exit 1
          fi
```

### 7.2 CodeBuild での ECR 統合

```yaml
# buildspec.yml
version: 0.2

env:
  variables:
    AWS_DEFAULT_REGION: ap-northeast-1
    ECR_REPO_NAME: my-app

phases:
  pre_build:
    commands:
      - echo Logging in to Amazon ECR...
      - ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"
      - aws ecr get-login-password | docker login --username AWS --password-stdin ${ECR_URI}
      - IMAGE_TAG=$(echo $CODEBUILD_RESOLVED_SOURCE_VERSION | head -c 8)
      - IMAGE_URI="${ECR_URI}/${ECR_REPO_NAME}"

  build:
    commands:
      - echo Build started on `date`
      - docker build -t ${IMAGE_URI}:${IMAGE_TAG} .
      - docker tag ${IMAGE_URI}:${IMAGE_TAG} ${IMAGE_URI}:latest

  post_build:
    commands:
      - echo Pushing the Docker image...
      - docker push ${IMAGE_URI}:${IMAGE_TAG}
      - docker push ${IMAGE_URI}:latest
      - echo Writing image definitions file...
      - printf '[{"name":"web","imageUri":"%s"}]' ${IMAGE_URI}:${IMAGE_TAG} > imagedefinitions.json

artifacts:
  files:
    - imagedefinitions.json
```

---

## 8. セキュリティベストプラクティス

### 8.1 イメージ署名 (Sigstore/Cosign)

```bash
# Cosign によるイメージ署名
# 1. キーペアの生成
cosign generate-key-pair

# 2. イメージへの署名
cosign sign --key cosign.key \
  ${ECR_URI}/${REPO_NAME}:${IMAGE_TAG}

# 3. 署名の検証
cosign verify --key cosign.pub \
  ${ECR_URI}/${REPO_NAME}:${IMAGE_TAG}

# 4. キーレス署名 (OIDC プロバイダ利用)
cosign sign --identity-token=$(gcloud auth print-identity-token) \
  ${ECR_URI}/${REPO_NAME}:${IMAGE_TAG}
```

### 8.2 SBOM (Software Bill of Materials) の生成

```bash
# Syft による SBOM 生成
syft ${ECR_URI}/${REPO_NAME}:${IMAGE_TAG} \
  -o spdx-json > sbom.spdx.json

# SBOM を ECR に OCI アーティファクトとしてアタッチ
cosign attach sbom \
  --sbom sbom.spdx.json \
  ${ECR_URI}/${REPO_NAME}:${IMAGE_TAG}

# Grype による SBOM ベースの脆弱性スキャン
grype sbom:sbom.spdx.json
```

### 8.3 ベースイメージの管理

```
ベースイメージ戦略:

推奨イメージ (軽量・セキュア):
  1. distroless (google) - シェルなし、最小限
  2. Alpine Linux - 5MB、musl libc
  3. Debian slim - glibc 互換、軽量
  4. Amazon Linux 2023 - AWS 最適化

非推奨:
  ❌ ubuntu:latest (大きい、頻繁に更新)
  ❌ node:latest (700MB+)
  ❌ python:latest (900MB+)

イメージサイズ比較:
  python:3.12         → ~900 MB
  python:3.12-slim    → ~120 MB
  python:3.12-alpine  → ~50 MB
  distroless/python3  → ~30 MB
```

---

## 9. アンチパターン

### 9.1 latest タグのみでの運用

```
[悪い例]
docker push my-app:latest  (毎回 latest を上書き)

問題:
  - ロールバック時にどのイメージに戻すか不明
  - 複数環境で異なるバージョンが latest として存在
  - タグの不変性を設定できない

[良い例]
docker push my-app:v1.2.3           (セマンティックバージョニング)
docker push my-app:git-abc123def    (Git SHA)
docker push my-app:build-456        (ビルド番号)

さらに良い例 (複数タグの併用):
docker push my-app:v1.2.3
docker push my-app:git-abc123def
docker push my-app:latest           (参考用、デプロイには使わない)
```

### 9.2 ライフサイクルポリシー未設定

**問題点**: イメージが蓄積し続け、ストレージコストが増大する。数千のイメージが残り、必要なイメージの特定が困難になる。

**改善**: プロジェクト初期からライフサイクルポリシーを設定し、untagged イメージの自動削除と、タグ付きイメージの保持数上限を定める。

### 9.3 ルートユーザーでのコンテナ実行

```
[悪い例]
FROM python:3.12
COPY . /app
CMD ["python", "/app/main.py"]
→ root (UID 0) で実行される

[良い例]
FROM python:3.12
RUN useradd --create-home appuser
USER appuser
COPY --chown=appuser:appuser . /app
CMD ["python", "/app/main.py"]
→ 非 root (UID 1000) で実行される

[さらに良い例]
FROM gcr.io/distroless/python3-debian12:nonroot
COPY . /app
CMD ["python", "/app/main.py"]
→ nonroot (UID 65532) で実行、シェルアクセスも不可
```

### 9.4 認証情報のハードコード

```
[悪い例]
FROM python:3.12
ENV DB_PASSWORD=my-secret-password  ← イメージに含まれる！
COPY . /app
CMD ["python", "/app/main.py"]

[良い例]
FROM python:3.12
# 認証情報は環境変数で実行時に注入
# ECS: タスク定義の secrets パラメータで Secrets Manager を参照
# EKS: Kubernetes Secrets + IRSA で管理
COPY . /app
CMD ["python", "/app/main.py"]
```

---

## 10. CloudFormation テンプレート

### 10.1 ECR リポジトリの完全構成テンプレート

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'ECR リポジトリ構成テンプレート'

Parameters:
  EnvironmentName:
    Type: String
    Default: prod
    AllowedValues: [dev, stg, prod]

  ProjectName:
    Type: String
    Default: my-app

Resources:
  # ECR リポジトリ
  ECRRepository:
    Type: AWS::ECR::Repository
    Properties:
      RepositoryName: !Sub '${ProjectName}'
      ImageScanningConfiguration:
        ScanOnPush: true
      ImageTagMutability: IMMUTABLE
      EncryptionConfiguration:
        EncryptionType: AES256
      LifecyclePolicy:
        LifecyclePolicyText: |
          {
            "rules": [
              {
                "rulePriority": 1,
                "description": "untagged images expire after 7 days",
                "selection": {
                  "tagStatus": "untagged",
                  "countType": "sinceImagePushed",
                  "countUnit": "days",
                  "countNumber": 7
                },
                "action": {"type": "expire"}
              },
              {
                "rulePriority": 2,
                "description": "Keep last 50 tagged images",
                "selection": {
                  "tagStatus": "tagged",
                  "tagPrefixList": ["v"],
                  "countType": "imageCountMoreThan",
                  "countNumber": 50
                },
                "action": {"type": "expire"}
              }
            ]
          }
      RepositoryPolicyText:
        Version: '2012-10-17'
        Statement:
          - Sid: AllowPullFromProdAccount
            Effect: Allow
            Principal:
              AWS: !Sub 'arn:aws:iam::${AWS::AccountId}:root'
            Action:
              - ecr:GetDownloadUrlForLayer
              - ecr:BatchGetImage
              - ecr:BatchCheckLayerAvailability
      Tags:
        - Key: Environment
          Value: !Ref EnvironmentName
        - Key: Project
          Value: !Ref ProjectName

  # スキャン通知用 EventBridge ルール
  ECRScanRule:
    Type: AWS::Events::Rule
    Properties:
      Name: !Sub '${ProjectName}-ecr-scan-alert'
      EventPattern:
        source:
          - aws.ecr
        detail-type:
          - ECR Image Scan
        detail:
          repository-name:
            - !Ref ECRRepository
          scan-status:
            - COMPLETE
      Targets:
        - Arn: !Ref AlertTopic
          Id: ECRScanAlert
          InputTransformer:
            InputPathsMap:
              repo: $.detail.repository-name
              tag: $.detail.image-tags[0]
              critical: $.detail.finding-severity-counts.CRITICAL
              high: $.detail.finding-severity-counts.HIGH
            InputTemplate: |
              "ECR Scan Alert: <repo>:<tag> - CRITICAL: <critical>, HIGH: <high>"

  # SNS トピック
  AlertTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: !Sub '${ProjectName}-ecr-alerts'

Outputs:
  RepositoryUri:
    Description: ECR リポジトリ URI
    Value: !GetAtt ECRRepository.RepositoryUri
    Export:
      Name: !Sub '${ProjectName}-ECRRepositoryUri'

  RepositoryArn:
    Description: ECR リポジトリ ARN
    Value: !GetAtt ECRRepository.Arn
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---

## 11. FAQ

### Q1. ECR の認証トークンの有効期限はどのくらいですか？

ECR の認証トークンは 12 時間有効である。CI/CD パイプラインでは、ビルドの最初に `aws ecr get-login-password` を実行してトークンを取得する。長時間稼働するビルドエージェントでは、cron 等で定期的に再認証が必要になる。

### Q2. ECR Public と Docker Hub の違いは何ですか？

ECR Public Gallery (public.ecr.aws) は AWS が提供するパブリックコンテナレジストリで、匿名プルが可能。Docker Hub と比較して、AWS アカウントからのプルはレート制限が緩和されている。AWS 公式のベースイメージ(Lambda, AL2023 等)は ECR Public で提供されている。

### Q3. イメージサイズを小さくするにはどうすべきですか？

マルチステージビルドでビルドツールを最終イメージから除外する。Alpine ベースの軽量イメージや distroless イメージを使用する。`.dockerignore` でビルドコンテキストから不要ファイルを除外する。レイヤーキャッシュを活用するため、変更頻度の低い命令(依存関係インストール)を先に記述する。

### Q4. ECR のプルスルーキャッシュとは何ですか？

Docker Hub、GitHub Container Registry、Quay.io などのアップストリームレジストリからのイメージプルを ECR 経由でキャッシュする機能である。Docker Hub のレート制限を回避し、プルの高速化とネットワークコスト削減を実現する。一度キャッシュされたイメージはその後 ECR から直接プルされる。

### Q5. マルチアーキテクチャイメージはどう管理すべきですか？

`docker buildx` を使用して amd64 と arm64 の両方のイメージを同一タグで管理できる。ECR はマニフェストリストをサポートしており、プル時にクライアントのアーキテクチャに適したイメージが自動的に選択される。Graviton (arm64) と x86_64 の両方で動作するアプリケーションでは、マルチアーキテクチャビルドを推奨する。

### Q6. イメージのダイジェスト (SHA) で管理すべきですか？

本番デプロイでは、タグではなくイメージダイジェスト (sha256:xxx) でイメージを指定することが最も安全である。タグは上書き可能 (MUTABLE の場合) だが、ダイジェストは不変である。ただし IMMUTABLE タグ設定であればタグでの管理も安全性が高い。CI/CD パイプラインではタグとダイジェストの両方を記録するのがベストプラクティスである。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| リポジトリ作成 | IMMUTABLE タグ + スキャン有効化が推奨 |
| イメージプッシュ | get-login-password で認証後、tag + push |
| ライフサイクルポリシー | untagged の自動削除、タグ付きの保持数制限 |
| イメージスキャン | ベーシック(無料) / 拡張(Inspector) |
| レプリケーション | クロスリージョン/クロスアカウントの自動複製 |
| プルスルーキャッシュ | Docker Hub レート制限の回避 |
| セキュリティ | 非 root 実行、マルチステージビルド、イメージ署名 |
| CI/CD 統合 | GitHub Actions / CodeBuild でのビルド・プッシュ自動化 |
| SBOM | ソフトウェア構成の可視化と脆弱性追跡 |

---

## 次に読むべきガイド

- [ECS 基礎](./00-ecs-basics.md) -- ECR イメージを ECS で実行する
- [EKS 概要](./02-eks-overview.md) -- ECR イメージを EKS で実行する
- [CodePipeline](../07-devops/02-codepipeline.md) -- ECR を CI/CD に統合する

---

## 参考文献

1. AWS 公式ドキュメント「Amazon ECR ユーザーガイド」 https://docs.aws.amazon.com/AmazonECR/latest/userguide/
2. AWS 公式「コンテナイメージのベストプラクティス」 https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/container-images.html
3. Docker 公式「Dockerfile ベストプラクティス」 https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
4. Sigstore/Cosign 公式ドキュメント https://docs.sigstore.dev/
5. AWS 公式「ECR プルスルーキャッシュ」 https://docs.aws.amazon.com/AmazonECR/latest/userguide/pull-through-cache.html

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

# Amazon ECR (Elastic Container Registry)

> コンテナイメージの保存・管理を行う Amazon ECR のリポジトリ作成、イメージのプッシュ/プル、ライフサイクルポリシー、イメージスキャンまでを体系的に学ぶ。

---

## この章で学ぶこと

1. **ECR リポジトリの作成と管理** -- プライベート/パブリックリポジトリの作成、アクセス制御の基本を理解する
2. **イメージのビルド・プッシュ・プル** -- Docker CLI を使ったイメージ操作と ECR 認証の仕組みを習得する
3. **ライフサイクルポリシーとイメージスキャン** -- 不要イメージの自動削除と脆弱性スキャンでセキュリティと運用コストを管理する

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
    }
  ]
}
```

```bash
# リポジトリポリシーの適用
aws ecr set-repository-policy \
  --repository-name my-app \
  --policy-text file://ecr-policy.json
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
      "description": "全タグ付きイメージは最新50個を保持",
      "selection": {
        "tagStatus": "tagged",
        "tagPrefixList": ["v"],
        "countType": "imageCountMoreThan",
        "countNumber": 50
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

# ポリシーのプレビュー (ドライラン)
aws ecr get-lifecycle-policy-preview \
  --repository-name my-app
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

### 5.2 スキャン結果の確認

```bash
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

    if critical > 0 or high > 0:
        message = {
            "text": f":warning: ECR スキャン結果: {repo}:{tag}\n"
                    f"CRITICAL: {critical}, HIGH: {high}\n"
                    f"確認してください。"
        }
        http = urllib3.PoolManager()
        http.request("POST", SLACK_WEBHOOK_URL,
                     body=json.dumps(message),
                     headers={"Content-Type": "application/json"})
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
# レプリケーション設定
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
            "region": "ap-northeast-1",
            "registryId": "987654321098"
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
```

---

## 7. アンチパターン

### 7.1 latest タグのみでの運用

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
```

### 7.2 ライフサイクルポリシー未設定

**問題点**: イメージが蓄積し続け、ストレージコストが増大する。数千のイメージが残り、必要なイメージの特定が困難になる。

**改善**: プロジェクト初期からライフサイクルポリシーを設定し、untagged イメージの自動削除と、タグ付きイメージの保持数上限を定める。

---

## 8. FAQ

### Q1. ECR の認証トークンの有効期限はどのくらいですか？

ECR の認証トークンは 12 時間有効である。CI/CD パイプラインでは、ビルドの最初に `aws ecr get-login-password` を実行してトークンを取得する。長時間稼働するビルドエージェントでは、cron 等で定期的に再認証が必要になる。

### Q2. ECR Public と Docker Hub の違いは何ですか？

ECR Public Gallery (public.ecr.aws) は AWS が提供するパブリックコンテナレジストリで、匿名プルが可能。Docker Hub と比較して、AWS アカウントからのプルはレート制限が緩和されている。AWS 公式のベースイメージ(Lambda, AL2023 等)は ECR Public で提供されている。

### Q3. イメージサイズを小さくするにはどうすべきですか？

マルチステージビルドでビルドツールを最終イメージから除外する。Alpine ベースの軽量イメージや distroless イメージを使用する。`.dockerignore` でビルドコンテキストから不要ファイルを除外する。レイヤーキャッシュを活用するため、変更頻度の低い命令(依存関係インストール)を先に記述する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| リポジトリ作成 | IMMUTABLE タグ + スキャン有効化が推奨 |
| イメージプッシュ | get-login-password で認証後、tag + push |
| ライフサイクルポリシー | untagged の自動削除、タグ付きの保持数制限 |
| イメージスキャン | ベーシック(無料) / 拡張(Inspector) |
| レプリケーション | クロスリージョン/クロスアカウントの自動複製 |
| セキュリティ | 非 root 実行、マルチステージビルド、イメージ署名 |

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

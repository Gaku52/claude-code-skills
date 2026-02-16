# Docker CI/CD

> GitHub Actionsを中心に、Dockerイメージのビルド自動化・テスト・レジストリプッシュ・デプロイパイプラインを構築する。

---

## この章で学ぶこと

1. **GitHub ActionsによるDockerイメージの自動ビルド・プッシュ**のワークフロー設計を理解する
2. **マルチプラットフォームビルドとキャッシュ戦略**による高速化手法を習得する
3. **ステージングから本番までのデプロイパイプライン**を構築できるようになる
4. **セキュリティスキャンとイメージ署名**をCI/CDに統合する手法を理解する
5. **GitLab CI / CircleCI**など他のCI/CDツールでのDocker連携パターンを把握する

---

## 1. Docker CI/CDパイプラインの全体像

### パイプラインアーキテクチャ

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Code   │    │  Build   │    │  Test    │    │  Deploy  │
│  Push   │───►│  Image   │───►│  Scan   │───►│  Release │
│         │    │          │    │  Verify  │    │          │
└─────────┘    └──────────┘    └──────────┘    └──────────┘
     │              │               │               │
     ▼              ▼               ▼               ▼
  git push     docker build    trivy scan     docker push
  PR作成       multi-stage     unit test      kubectl apply
  tag作成      layer cache     integration    docker compose
```

### CI/CDパイプラインの原則

```
┌─────────────────────────────────────────────────────────┐
│              CI/CDパイプラインの5原則                      │
│                                                          │
│  1. 再現可能性    同じコミットから常に同じイメージを生成   │
│  2. 不変性        ビルド済みイメージは変更しない          │
│  3. 高速性        キャッシュとパラレル実行で最適化        │
│  4. 安全性        シークレット管理、スキャン、署名        │
│  5. 可観測性      ビルドログ、メトリクス、通知の統合      │
└─────────────────────────────────────────────────────────┘
```

### CI/CDツール比較表

| ツール | Docker連携 | 特徴 | 無料枠 |
|--------|-----------|------|--------|
| GitHub Actions | Docker公式Action | GitHub統合、GHCR連携 | 2,000分/月 |
| GitLab CI | Docker-in-Docker | 組み込みレジストリ | 400分/月 |
| CircleCI | Docker Executor | 高速、Docker Layer Cache | 6,000分/月 |
| AWS CodeBuild | ECR連携 | AWSネイティブ | 100分/月 |
| Jenkins | Docker Plugin | 自己ホスト、高カスタマイズ性 | 無制限（自己ホスト） |

---

## 2. GitHub Actions基本構成

### コード例1: 基本的なDockerビルド・プッシュ

```yaml
# .github/workflows/docker-build.yml
name: Docker Build and Push

on:
  push:
    branches: [main, develop]
    tags: ["v*"]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      # 1. チェックアウト
      - name: Checkout repository
        uses: actions/checkout@v4

      # 2. Docker Buildxのセットアップ
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      # 3. レジストリへのログイン
      - name: Log in to Container Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      # 4. メタデータの抽出（タグ、ラベル）
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            # ブランチ名タグ
            type=ref,event=branch
            # PRナンバータグ
            type=ref,event=pr
            # セマンティックバージョニング
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}}
            # Git SHA（短縮）
            type=sha,prefix=sha-
            # latest（mainブランチのみ）
            type=raw,value=latest,enable={{is_default_branch}}

      # 5. ビルド & プッシュ
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
```

### タグ戦略のフロー

```
git push main
    └──► ghcr.io/user/app:main
         ghcr.io/user/app:sha-abc1234
         ghcr.io/user/app:latest

git push develop
    └──► ghcr.io/user/app:develop
         ghcr.io/user/app:sha-def5678

git tag v1.2.3
    └──► ghcr.io/user/app:1.2.3
         ghcr.io/user/app:1.2
         ghcr.io/user/app:1
         ghcr.io/user/app:sha-ghi9012
         ghcr.io/user/app:latest

Pull Request #42
    └──► ghcr.io/user/app:pr-42  (プッシュされない)
```

### タグ戦略の比較

| 戦略 | 例 | 用途 | 特徴 |
|------|---|------|------|
| セマンティックバージョン | v1.2.3 | 本番リリース | 人間が読みやすい |
| Git SHA | sha-abc1234 | 全ビルド | 完全な追跡可能性 |
| ブランチ名 | main, develop | 開発・ステージング | 自動更新される |
| タイムスタンプ | 20240115-1030 | CI/CD内部 | 時系列順序が明確 |
| latest | latest | 開発用途のみ | **本番で使用禁止** |

---

## 3. テスト統合

### コード例2: テスト・セキュリティスキャン統合パイプライン

```yaml
# .github/workflows/ci-pipeline.yml
name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # === ユニットテスト ===
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run unit tests in Docker
        run: |
          docker compose -f docker-compose.test.yml run --rm \
            --build \
            test npm run test:ci

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: coverage/

  # === Lint & 静的解析 ===
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Lint Dockerfile
        uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
          failure-threshold: warning

      - name: Lint docker-compose files
        run: |
          docker compose -f docker-compose.yml config -q
          docker compose -f docker-compose.prod.yml config -q

  # === イメージビルド ===
  build:
    needs: [test, lint]
    runs-on: ubuntu-latest
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
      image-tag: ${{ steps.meta.outputs.version }}
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=sha,prefix=sha-

      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          provenance: true
          sbom: true

  # === セキュリティスキャン ===
  security-scan:
    needs: [build]
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ghcr.io/${{ github.repository }}:${{ github.sha }}
          format: "sarif"
          output: "trivy-results.sarif"
          severity: "CRITICAL,HIGH"
          exit-code: "1"  # CRITICAL/HIGH が見つかったら失敗

      - name: Upload Trivy scan results
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: "trivy-results.sarif"

  # === Dockerfile ベストプラクティスチェック ===
  dockerfile-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Dockle
        uses: erzz/dockle-action@v1
        with:
          image: ghcr.io/${{ github.repository }}:${{ github.sha }}
          exit-code: "1"
          exit-level: "WARN"

  # === 統合テスト ===
  integration-test:
    needs: [build]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run integration tests
        env:
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker compose -f docker-compose.integration.yml up -d

          # ヘルスチェック待ち
          for i in $(seq 1 30); do
            if docker compose -f docker-compose.integration.yml exec -T api \
              wget -q --spider http://localhost:8080/health 2>/dev/null; then
              echo "Service is healthy"
              break
            fi
            echo "Waiting for services... ($i/30)"
            sleep 2
          done

          # テスト実行
          docker compose -f docker-compose.integration.yml run --rm \
            test npm run test:integration

          # クリーンアップ
          docker compose -f docker-compose.integration.yml down -v
```

```yaml
# docker-compose.test.yml
version: "3.9"

services:
  test:
    build:
      context: .
      target: test  # テスト用ステージ
    volumes:
      - ./coverage:/app/coverage
    environment:
      NODE_ENV: test
      DATABASE_URL: postgres://test:test@db:5432/testdb

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
      POSTGRES_DB: testdb
    tmpfs:
      - /var/lib/postgresql/data  # テストはメモリ上で高速実行
```

```yaml
# docker-compose.integration.yml
version: "3.9"

services:
  api:
    image: ghcr.io/${GITHUB_REPOSITORY}:${IMAGE_TAG}
    environment:
      NODE_ENV: test
      DATABASE_URL: postgres://test:test@db:5432/testdb
      REDIS_URL: redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:8080/health"]
      interval: 5s
      timeout: 3s
      retries: 10

  test:
    build:
      context: .
      target: test
    environment:
      API_URL: http://api:8080
      NODE_ENV: test
    depends_on:
      api:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
      POSTGRES_DB: testdb
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test"]
      interval: 5s
      timeout: 3s
      retries: 5
    tmpfs:
      - /var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

---

## 4. キャッシュ戦略

### コード例3: 高度なキャッシュ設定

```yaml
# .github/workflows/cached-build.yml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      # 方式1: GitHub Actions Cache（推奨）
      - name: Build with GHA cache
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

      # 方式2: レジストリキャッシュ
      # cache-from: type=registry,ref=ghcr.io/${{ github.repository }}:buildcache
      # cache-to: type=registry,ref=ghcr.io/${{ github.repository }}:buildcache,mode=max

      # 方式3: ローカルキャッシュ
      # cache-from: type=local,src=/tmp/.buildx-cache
      # cache-to: type=local,dest=/tmp/.buildx-cache-new,mode=max
```

### キャッシュ方式の比較表

| 方式 | 速度 | 容量制限 | CI間共有 | 設定の簡便さ | コスト |
|------|------|---------|---------|-------------|--------|
| GHA Cache | 高速 | 10GB | 同一リポ | 最も簡単 | 無料 |
| Registry Cache | 中速 | 無制限 | 全環境 | 中程度 | レジストリ料金 |
| Local Cache | 最速 | ディスク依存 | 不可 | 簡単 | 無料 |
| Inline Cache | 中速 | イメージ内 | 全環境 | 簡単 | 無料 |
| S3 Cache | 中速 | 無制限 | 全環境 | やや複雑 | S3料金 |

### キャッシュの動作原理

```
初回ビルド（キャッシュなし）
┌──────────────────────────────────┐
│ Layer 1: FROM node:20-alpine     │  ← ダウンロード
│ Layer 2: COPY package*.json      │  ← 新規作成
│ Layer 3: RUN npm ci              │  ← 新規作成（遅い）
│ Layer 4: COPY . .                │  ← 新規作成
│ Layer 5: RUN npm run build       │  ← 新規作成
└──────────────────────────────────┘
  合計: 3分

2回目ビルド（ソースコードのみ変更）
┌──────────────────────────────────┐
│ Layer 1: FROM node:20-alpine     │  ← キャッシュHIT
│ Layer 2: COPY package*.json      │  ← キャッシュHIT
│ Layer 3: RUN npm ci              │  ← キャッシュHIT ★高速
│ Layer 4: COPY . .                │  ← 再作成（変更検知）
│ Layer 5: RUN npm run build       │  ← 再作成
└──────────────────────────────────┘
  合計: 30秒
```

### Dockerfileのキャッシュ最適化

```dockerfile
# === キャッシュを最大限活用するDockerfile ===
FROM node:20-alpine AS builder

WORKDIR /app

# 1. パッケージマネージャーのロックファイルだけ先にコピー
# → 依存関係が変わらない限りこのレイヤーはキャッシュされる
COPY package.json package-lock.json ./

# 2. 依存関係インストール（最も遅いステップ）
# → ロックファイルが変わった時だけ再実行
RUN --mount=type=cache,target=/root/.npm \
    npm ci

# 3. ソースコードをコピー（頻繁に変わる）
COPY tsconfig.json ./
COPY src/ ./src/

# 4. ビルド
RUN npm run build

# === 本番ステージ ===
FROM node:20-alpine AS production

WORKDIR /app

# 本番依存関係のみインストール
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production

# ビルド成果物をコピー
COPY --from=builder /app/dist ./dist

USER node
CMD ["node", "dist/server.js"]
```

### BuildKit マウントキャッシュ

```dockerfile
# BuildKit のキャッシュマウント（--mount=type=cache）を活用
# パッケージマネージャーのキャッシュをビルド間で共有

# Go
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    go build -o /app/server ./cmd/server

# Python
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Rust
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release

# Java/Maven
RUN --mount=type=cache,target=/root/.m2 \
    mvn package -DskipTests

# Java/Gradle
RUN --mount=type=cache,target=/root/.gradle \
    gradle build -x test
```

---

## 5. デプロイパイプライン

### コード例4: ステージング→本番デプロイ

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    tags: ["v*"]

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
      version: ${{ steps.version.outputs.version }}
      digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@v4

      - name: Extract version
        id: version
        run: echo "version=${GITHUB_REF_NAME#v}" >> $GITHUB_OUTPUT

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.version.outputs.version }}
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
          provenance: true
          sbom: true

  # === セキュリティスキャン ===
  security-scan:
    needs: [build]
    runs-on: ubuntu-latest
    steps:
      - name: Run Trivy
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ needs.build.outputs.version }}
          severity: "CRITICAL"
          exit-code: "1"

  # === ステージングデプロイ ===
  deploy-staging:
    needs: [build, security-scan]
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://staging.example.com
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to staging
        env:
          VERSION: ${{ needs.build.outputs.version }}
        run: |
          # SSH経由でデプロイ
          ssh -o StrictHostKeyChecking=no deploy@staging.example.com << EOF
            cd /opt/app
            export VERSION=${VERSION}
            docker compose pull
            docker compose up -d --remove-orphans
            docker compose exec -T api wget -q --spider http://localhost:8080/health
          EOF

      - name: Run smoke tests
        run: |
          sleep 10
          curl -f https://staging.example.com/health || exit 1
          curl -f https://staging.example.com/api/status || exit 1

      - name: Run E2E tests
        run: |
          docker run --rm \
            -e BASE_URL=https://staging.example.com \
            my-e2e-tests:latest \
            npm run test:e2e

  # === 本番デプロイ（手動承認後） ===
  deploy-production:
    needs: [deploy-staging]
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://www.example.com
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to production
        env:
          VERSION: ${{ needs.build.outputs.version }}
        run: |
          ssh -o StrictHostKeyChecking=no deploy@prod.example.com << 'EOF'
            cd /opt/app

            # ローリングデプロイ
            export VERSION=${{ env.VERSION }}
            docker compose pull
            docker compose up -d --remove-orphans --scale api=3

            # ヘルスチェック確認
            for i in $(seq 1 30); do
              if docker compose exec -T api wget -q --spider http://localhost:8080/health; then
                echo "Health check passed"
                break
              fi
              echo "Waiting for health check... ($i/30)"
              sleep 2
            done

            # 古いイメージの削除
            docker image prune -af --filter "until=168h"
          EOF

      - name: Verify deployment
        run: |
          curl -f https://www.example.com/health
          curl -f https://www.example.com/api/status

      - name: Notify deployment
        if: success()
        uses: slackapi/slack-github-action@v1.24.0
        with:
          channel-id: "#deployments"
          slack-message: "Deployed v${{ needs.build.outputs.version }} to production"
        env:
          SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}
```

### デプロイフロー

```
git tag v1.2.3 && git push --tags
    │
    ▼
┌──────────┐     ┌─────────────┐     ┌──────────────┐
│  Build   │────►│  Security   │────►│  Staging     │
│  & Push  │     │  Scan       │     │  Deploy      │
│          │     │             │     │  (自動)       │
└──────────┘     └─────────────┘     └──────┬───────┘
                                             │
                                        Smoke Test
                                        E2E Test
                                             │
                                    ┌────────▼───────┐
                                    │  Production    │
                                    │  Deploy        │
                                    │  (手動承認)     │
                                    └────────┬───────┘
                                             │
                                        Health Check
                                        ローリング更新
                                        Slack通知
```

### ロールバック戦略

```yaml
# .github/workflows/rollback.yml
name: Rollback Production

on:
  workflow_dispatch:
    inputs:
      version:
        description: "Version to rollback to (e.g., 1.2.2)"
        required: true

jobs:
  rollback:
    runs-on: ubuntu-latest
    environment:
      name: production
    steps:
      - uses: actions/checkout@v4

      - name: Verify image exists
        run: |
          docker pull ghcr.io/${{ github.repository }}:${{ inputs.version }}

      - name: Rollback production
        run: |
          ssh deploy@prod.example.com << EOF
            cd /opt/app
            export VERSION=${{ inputs.version }}
            docker compose pull
            docker compose up -d --remove-orphans

            # ヘルスチェック
            for i in $(seq 1 30); do
              if docker compose exec -T api wget -q --spider http://localhost:8080/health; then
                echo "Rollback successful - v${{ inputs.version }}"
                break
              fi
              sleep 2
            done
          EOF

      - name: Notify rollback
        uses: slackapi/slack-github-action@v1.24.0
        with:
          channel-id: "#deployments"
          slack-message: "ROLLBACK: Production rolled back to v${{ inputs.version }}"
        env:
          SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}
```

---

## 6. マルチプラットフォームビルド

### コード例5: ARM64 + AMD64 マルチプラットフォーム

```yaml
# .github/workflows/multi-platform.yml
name: Multi-Platform Build

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
        with:
          platforms: linux/amd64,linux/arm64

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push multi-platform
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          platforms: linux/amd64,linux/arm64
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### プラットフォーム別ビルドのマトリックス戦略

```yaml
# 高速化: プラットフォームごとに並列ビルドし、後でマニフェストを統合
jobs:
  build-platform:
    strategy:
      matrix:
        platform:
          - linux/amd64
          - linux/arm64
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push by digest
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: ${{ matrix.platform }}
          outputs: type=image,name=ghcr.io/${{ github.repository }},push-by-digest=true,name-canonical=true,push=true
          cache-from: type=gha,scope=${{ matrix.platform }}
          cache-to: type=gha,scope=${{ matrix.platform }},mode=max

      - name: Export digest
        run: echo "${{ steps.build.outputs.digest }}" > /tmp/digest-${{ strategy.job-index }}

      - uses: actions/upload-artifact@v4
        with:
          name: digest-${{ strategy.job-index }}
          path: /tmp/digest-*

  # マニフェストリストの作成
  merge:
    needs: [build-platform]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: digest-*
          merge-multiple: true
          path: /tmp/digests

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Create manifest list
        run: |
          digests=$(cat /tmp/digests/digest-*)
          docker buildx imagetools create \
            -t ghcr.io/${{ github.repository }}:latest \
            $digests
```

---

## 7. Docker Compose によるローカルCI再現

### コード例6: ローカルで CI パイプラインを再現

```yaml
# docker-compose.ci.yml
version: "3.9"

services:
  lint:
    image: hadolint/hadolint:latest-alpine
    volumes:
      - ./Dockerfile:/Dockerfile:ro
    command: hadolint /Dockerfile

  test:
    build:
      context: .
      target: test
    command: npm run test:ci
    environment:
      NODE_ENV: test
      DATABASE_URL: postgres://ci:ci@db:5432/ci_test
    depends_on:
      db:
        condition: service_healthy

  security-scan:
    image: aquasec/trivy:latest
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - trivy-cache:/root/.cache/
    command: image --severity HIGH,CRITICAL my-app:test

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ci
      POSTGRES_PASSWORD: ci
      POSTGRES_DB: ci_test
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ci"]
      interval: 5s
      timeout: 3s
      retries: 5
    tmpfs:
      - /var/lib/postgresql/data

volumes:
  trivy-cache:
```

```bash
# ローカルでCIパイプラインを実行
docker compose -f docker-compose.ci.yml run --rm lint
docker compose -f docker-compose.ci.yml run --rm test
docker compose -f docker-compose.ci.yml run --rm security-scan
docker compose -f docker-compose.ci.yml down -v
```

### Makefile によるCI/CDタスク管理

```makefile
# Makefile - CI/CDタスクの統一インターフェース
.PHONY: build test lint scan deploy-staging deploy-production

# 変数
IMAGE_NAME := ghcr.io/myorg/myapp
VERSION := $(shell git describe --tags --always)

# ビルド
build:
	docker build -t $(IMAGE_NAME):$(VERSION) -t $(IMAGE_NAME):latest .

# テスト
test:
	docker compose -f docker-compose.test.yml run --rm --build test

# Lint
lint:
	docker run --rm -v $(PWD)/Dockerfile:/Dockerfile \
		hadolint/hadolint:latest-alpine hadolint /Dockerfile

# セキュリティスキャン
scan:
	trivy image --severity HIGH,CRITICAL $(IMAGE_NAME):$(VERSION)

# 全CIステップ実行
ci: lint test build scan

# ステージングデプロイ
deploy-staging:
	VERSION=$(VERSION) docker compose -f docker-compose.staging.yml pull
	VERSION=$(VERSION) docker compose -f docker-compose.staging.yml up -d

# 本番デプロイ
deploy-production:
	@echo "Deploying $(VERSION) to production..."
	VERSION=$(VERSION) docker compose -f docker-compose.prod.yml pull
	VERSION=$(VERSION) docker compose -f docker-compose.prod.yml up -d --remove-orphans

# クリーンアップ
clean:
	docker compose -f docker-compose.test.yml down -v
	docker image prune -f
```

---

## 8. GitLab CI / CircleCI での Docker CI/CD

### GitLab CI の Docker ビルド

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - scan
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE
  DOCKER_TAG: $CI_COMMIT_SHORT_SHA

# テスト
test:
  stage: test
  image: docker:24-dind
  services:
    - docker:24-dind
  script:
    - docker compose -f docker-compose.test.yml run --rm test

# ビルド & プッシュ
build:
  stage: build
  image: docker:24-dind
  services:
    - docker:24-dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $DOCKER_IMAGE:$DOCKER_TAG -t $DOCKER_IMAGE:latest .
    - docker push $DOCKER_IMAGE:$DOCKER_TAG
    - docker push $DOCKER_IMAGE:latest

# セキュリティスキャン
scan:
  stage: scan
  image: aquasec/trivy:latest
  script:
    - trivy image --severity CRITICAL,HIGH $DOCKER_IMAGE:$DOCKER_TAG

# ステージングデプロイ
deploy-staging:
  stage: deploy
  environment:
    name: staging
    url: https://staging.example.com
  script:
    - ssh deploy@staging.example.com "cd /opt/app && VERSION=$DOCKER_TAG docker compose up -d"
  only:
    - main

# 本番デプロイ（手動）
deploy-production:
  stage: deploy
  environment:
    name: production
    url: https://www.example.com
  script:
    - ssh deploy@prod.example.com "cd /opt/app && VERSION=$DOCKER_TAG docker compose up -d"
  when: manual
  only:
    - tags
```

### CircleCI の Docker ビルド

```yaml
# .circleci/config.yml
version: 2.1

orbs:
  docker: circleci/docker@2.4.0

executors:
  docker-executor:
    docker:
      - image: cimg/base:2024.01

jobs:
  build-and-push:
    executor: docker-executor
    steps:
      - checkout
      - setup_remote_docker:
          docker_layer_caching: true  # DLC（有料機能）
      - docker/check:
          registry: ghcr.io
          docker-username: GHCR_USER
          docker-password: GHCR_TOKEN
      - docker/build:
          image: $CIRCLE_PROJECT_USERNAME/$CIRCLE_PROJECT_REPONAME
          registry: ghcr.io
          tag: ${CIRCLE_SHA1:0:8},latest
      - docker/push:
          image: $CIRCLE_PROJECT_USERNAME/$CIRCLE_PROJECT_REPONAME
          registry: ghcr.io
          tag: ${CIRCLE_SHA1:0:8},latest

  security-scan:
    docker:
      - image: aquasec/trivy:latest
    steps:
      - run:
          name: Scan image
          command: |
            trivy image --severity CRITICAL,HIGH \
              ghcr.io/$CIRCLE_PROJECT_USERNAME/$CIRCLE_PROJECT_REPONAME:${CIRCLE_SHA1:0:8}

workflows:
  build-deploy:
    jobs:
      - build-and-push:
          context: docker-credentials
      - security-scan:
          requires:
            - build-and-push
```

---

## 9. イメージ署名とサプライチェーンセキュリティ

### Cosign によるイメージ署名

```yaml
# GitHub Actions でのイメージ署名
- name: Install Cosign
  uses: sigstore/cosign-installer@v3

- name: Sign the image
  env:
    COSIGN_EXPERIMENTAL: "1"
  run: |
    cosign sign --yes \
      ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}

- name: Verify the signature
  run: |
    cosign verify \
      --certificate-identity "https://github.com/${{ github.repository }}/.github/workflows/docker-build.yml@refs/heads/main" \
      --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
      ghcr.io/${{ github.repository }}:latest
```

### SBOM（ソフトウェア部品表）の生成

```yaml
# ビルド時にSBOMを自動生成
- name: Build with SBOM
  uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    tags: ghcr.io/${{ github.repository }}:latest
    sbom: true        # BuildKit によるSBOM生成
    provenance: true  # SLSA Provenance の付与

# または Syft で明示的にSBOM生成
- name: Generate SBOM with Syft
  uses: anchore/sbom-action@v0
  with:
    image: ghcr.io/${{ github.repository }}:latest
    format: spdx-json
    output-file: sbom.spdx.json

- name: Upload SBOM
  uses: actions/upload-artifact@v4
  with:
    name: sbom
    path: sbom.spdx.json
```

---

## 10. AWS ECR / Docker Hub へのデプロイ

### AWS ECR へのプッシュ

```yaml
# .github/workflows/ecr-push.yml
jobs:
  build-push-ecr:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions-ecr
          aws-region: ap-northeast-1

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push to ECR
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ steps.login-ecr.outputs.registry }}/my-app:${{ github.sha }}
            ${{ steps.login-ecr.outputs.registry }}/my-app:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### Docker Hub へのプッシュ

```yaml
# .github/workflows/dockerhub-push.yml
jobs:
  build-push-dockerhub:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push to Docker Hub
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            myorg/my-app:${{ github.sha }}
            myorg/my-app:latest
```

---

## アンチパターン

### アンチパターン1: latest タグのみでのデプロイ

```yaml
# NG: latestタグだけでデプロイ
services:
  app:
    image: my-app:latest  # どのバージョンが動いているか不明

# OK: 明示的なバージョンタグ
services:
  app:
    image: my-app:1.2.3   # 完全なバージョン指定
    # または
    image: my-app:sha-abc1234  # Git SHA で特定
```

**なぜ問題か**: `latest` タグはミュータブル（上書き可能）であり、どのコミットのコードが本番で動いているか追跡できない。ロールバックも困難。

### アンチパターン2: CI上でのシークレットのハードコード

```yaml
# NG: ワークフロー内にシークレットを直書き
- name: Login to Docker Hub
  run: docker login -u myuser -p MyP@ssw0rd!

# OK: GitHub Secretsを使用
- name: Login to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}
```

**なぜ問題か**: リポジトリにシークレットが漏洩し、認証情報が第三者に悪用される。GitHub Secretsは暗号化されてログにもマスクされる。

### アンチパターン3: テストなしでのデプロイ

```yaml
# NG: ビルドしたら即デプロイ
jobs:
  build-and-deploy:
    steps:
      - uses: docker/build-push-action@v5
      - run: ssh prod "docker pull && docker compose up -d"

# OK: テスト→スキャン→ステージング→承認→本番
jobs:
  test: ...
  build: { needs: [test] }
  scan: { needs: [build] }
  deploy-staging: { needs: [scan] }
  deploy-production: { needs: [deploy-staging] }
```

**なぜ問題か**: テストやセキュリティスキャンをスキップすると、バグや脆弱性が本番に到達する。ステージングでの検証を経ることで、本番障害のリスクを低減する。

### アンチパターン4: ビルドとデプロイの密結合

```yaml
# NG: 1つのジョブ内でビルドからデプロイまで実行
jobs:
  all-in-one:
    steps:
      - run: docker build .
      - run: docker push
      - run: ssh prod "deploy"

# OK: ステージごとに分離し、ゲートを設ける
jobs:
  build: ...
  test: { needs: [build] }
  deploy: { needs: [test], environment: production }
```

**なぜ問題か**: 密結合すると、テスト失敗時にもデプロイが実行されるリスクがある。また、同じイメージを複数環境にデプロイする際に再ビルドが必要になる。

---

## FAQ

### Q1: Docker Hub と GitHub Container Registry (GHCR) のどちらを使うべき？

**GHCR推奨**: GitHub Actionsとの連携がシームレス（`GITHUB_TOKEN` で認証可能）、リポジトリの可視性と連動、無料枠が十分。Docker Hubはパブリックイメージの配布に適するが、プルレート制限（100回/6時間）がCI環境で問題になることがある。

### Q2: CI上でのDockerビルドが遅い場合の対策は？

1. **レイヤーキャッシュ**: `cache-from: type=gha` を設定
2. **マルチステージビルド**: テスト用ステージと本番ステージを分離
3. **依存関係の分離**: `package.json` を先にCOPYし、`npm ci` のレイヤーをキャッシュ
4. **BuildKitマウントキャッシュ**: `--mount=type=cache` でパッケージキャッシュを共有
5. **並列ビルド**: 独立したサービスは `matrix` 戦略で並列実行
6. **ランナースペック向上**: `runs-on: ubuntu-latest-8-cores` など大型ランナーを使用

### Q3: ロールバックはどうやって行う？

```bash
# 即座に前のバージョンに戻す
docker compose pull  # 旧バージョンタグに切り替え
VERSION=1.2.2 docker compose up -d

# または特定のSHAに戻す
docker compose up -d --no-deps \
  -e IMAGE_TAG=sha-abc1234 \
  api
```

タグを使ったイミュータブルなデプロイを行うことで、任意のバージョンへの即座のロールバックが可能になる。

### Q4: GitHub Actions の GITHUB_TOKEN でGHCRにプッシュできないときは？

以下を確認する:
1. ワークフローの `permissions` で `packages: write` を設定しているか
2. リポジトリの Settings > Actions > General > Workflow permissions が "Read and write permissions" になっているか
3. Organization の場合、パッケージの可視性設定が正しいか

### Q5: モノレポでの Docker CI/CD はどう設計するか？

```yaml
# パスフィルターで変更があったサービスのみビルド
on:
  push:
    paths:
      - "services/api/**"
      - "shared/**"

# または matrix 戦略で全サービスを並列ビルド
jobs:
  build:
    strategy:
      matrix:
        service: [api, worker, frontend]
    steps:
      - name: Build ${{ matrix.service }}
        uses: docker/build-push-action@v5
        with:
          context: ./services/${{ matrix.service }}
          tags: ghcr.io/${{ github.repository }}/${{ matrix.service }}:${{ github.sha }}
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| GitHub Actions | Docker公式Actionで統一。GHCR連携が最も簡便 |
| タグ戦略 | セマンティックバージョニング + Git SHA。latestだけに依存しない |
| キャッシュ | GHA Cacheが推奨。レイヤーの順序最適化で効果最大化 |
| セキュリティ | Trivyスキャン、Hadolint、GitHub Secretsを必ず使用 |
| テスト | Docker Compose でテスト環境を再現。CI とローカルで同一 |
| デプロイ | ステージング→承認→本番のゲート付きパイプライン |
| ロールバック | イミュータブルタグで即座にロールバック可能 |
| イメージ署名 | Cosign でイメージの真正性を保証 |
| SBOM | サプライチェーンの透明性確保 |
| マルチプラットフォーム | QEMU + Buildx で ARM64/AMD64 対応 |

---

## 次に読むべきガイド

- [オーケストレーション概要](../05-orchestration/00-orchestration-overview.md) -- K8s/Swarmへのデプロイ拡張
- [コンテナセキュリティ](../06-security/00-container-security.md) -- CIでのイメージスキャン強化
- [サプライチェーンセキュリティ](../06-security/01-supply-chain-security.md) -- イメージ署名とSBOM

---

## 参考文献

1. GitHub Actions 公式ドキュメント "Building and testing containers" -- https://docs.github.com/en/actions/publishing-packages/publishing-docker-images
2. Docker 公式 GitHub Actions -- https://github.com/docker/build-push-action
3. Docker 公式ドキュメント "CI/CD best practices" -- https://docs.docker.com/build/ci/github-actions/
4. Hadolint (Dockerfile Linter) -- https://github.com/hadolint/hadolint
5. Aqua Security Trivy -- https://github.com/aquasecurity/trivy
6. Sigstore Cosign -- https://github.com/sigstore/cosign
7. SLSA (Supply chain Levels for Software Artifacts) -- https://slsa.dev/
8. Docker 公式ドキュメント "BuildKit" -- https://docs.docker.com/build/buildkit/

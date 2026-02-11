# Docker CI/CD

> GitHub Actionsを中心に、Dockerイメージのビルド自動化・テスト・レジストリプッシュ・デプロイパイプラインを構築する。

---

## この章で学ぶこと

1. **GitHub ActionsによるDockerイメージの自動ビルド・プッシュ**のワークフロー設計を理解する
2. **マルチプラットフォームビルドとキャッシュ戦略**による高速化手法を習得する
3. **ステージングから本番までのデプロイパイプライン**を構築できるようになる

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

### CI/CDツール比較表

| ツール | Docker連携 | 特徴 | 無料枠 |
|--------|-----------|------|--------|
| GitHub Actions | Docker公式Action | GitHub統合、GHCR連携 | 2,000分/月 |
| GitLab CI | Docker-in-Docker | 組み込みレジストリ | 400分/月 |
| CircleCI | Docker Executor | 高速、Docker Layer Cache | 6,000分/月 |
| AWS CodeBuild | ECR連携 | AWSネイティブ | 100分/月 |

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

  # === イメージビルド ===
  build:
    needs: [test, lint]
    runs-on: ubuntu-latest
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
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

      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

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

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: "trivy-results.sarif"

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
          docker compose -f docker-compose.integration.yml run --rm \
            test npm run test:integration
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

| 方式 | 速度 | 容量制限 | CI間共有 | 設定の簡便さ |
|------|------|---------|---------|-------------|
| GHA Cache | 高速 | 10GB | 同一リポ | 最も簡単 |
| Registry Cache | 中速 | 無制限 | 全環境 | 中程度 |
| Local Cache | 最速 | ディスク依存 | 不可 | 簡単 |
| Inline Cache | 中速 | イメージ内 | 全環境 | 簡単 |

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
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.version.outputs.version }}
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # === ステージングデプロイ ===
  deploy-staging:
    needs: [build]
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
```

### デプロイフロー

```
git tag v1.2.3 && git push --tags
    │
    ▼
┌──────────┐     ┌─────────────┐     ┌──────────────┐
│  Build   │────►│  Staging    │────►│  Production  │
│  & Push  │     │  Deploy     │     │  Deploy      │
│          │     │  (自動)      │     │  (手動承認)   │
└──────────┘     └──────┬──────┘     └──────┬───────┘
                        │                    │
                   Smoke Test           Health Check
                   自動実行              ローリング更新
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

---

## FAQ

### Q1: Docker Hub と GitHub Container Registry (GHCR) のどちらを使うべき？

**GHCR推奨**: GitHub Actionsとの連携がシームレス（`GITHUB_TOKEN` で認証可能）、リポジトリの可視性と連動、無料枠が十分。Docker Hubはパブリックイメージの配布に適するが、プルレート制限（100回/6時間）がCI環境で問題になることがある。

### Q2: CI上でのDockerビルドが遅い場合の対策は？

1. **レイヤーキャッシュ**: `cache-from: type=gha` を設定
2. **マルチステージビルド**: テスト用ステージと本番ステージを分離
3. **依存関係の分離**: `package.json` を先にCOPYし、`npm ci` のレイヤーをキャッシュ
4. **Buildxのインラインキャッシュ**: `BUILDKIT_INLINE_CACHE=1`
5. **並列ビルド**: 独立したサービスは `matrix` 戦略で並列実行

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

# イメージ管理

> Docker イメージの取得・作成・配布からレジストリの活用まで、イメージのライフサイクル全体を管理するための実践ガイド。

---

## この章で学ぶこと

1. **Docker イメージのレイヤー構造とタグ体系**を理解し、効率的なイメージ管理ができる
2. **Docker Hub と GitHub Container Registry** の使い分けを理解し、イメージを配布できる
3. **イメージの検査・セキュリティ確認・クリーンアップ**を実践できる
4. **マルチプラットフォームビルド**を理解し、異なるアーキテクチャ向けのイメージを構築・配布できる
5. **CI/CD パイプラインにおけるイメージ管理戦略**を設計し、自動化された安全なイメージ配布フローを構築できる

---

## 1. イメージの構造

### 1.1 レイヤーモデル

```
+------------------------------------------------------+
|              Docker イメージのレイヤー構造              |
|                                                      |
|  +------------------------------------------------+ |
|  |  Layer 4: COPY . /app  (アプリコード)            | |  <- 変更頻度: 高
|  +------------------------------------------------+ |
|  |  Layer 3: RUN npm install (依存パッケージ)       | |  <- 変更頻度: 中
|  +------------------------------------------------+ |
|  |  Layer 2: RUN apt-get install (システムパッケージ)| |  <- 変更頻度: 低
|  +------------------------------------------------+ |
|  |  Layer 1: FROM node:20-alpine (ベースイメージ)   | |  <- 変更頻度: 最低
|  +------------------------------------------------+ |
|                                                      |
|  各レイヤーは SHA256 ハッシュで識別される               |
|  変更されていないレイヤーはキャッシュから再利用           |
+------------------------------------------------------+
```

Docker イメージは Union File System（OverlayFS）を使って複数の読み取り専用レイヤーを重ね合わせた構造になっている。各レイヤーは前のレイヤーからの差分（変更分）のみを保持するため、共通のベースイメージを使うイメージ同士ではレイヤーの共有が行われ、ディスク使用量とダウンロード時間を大幅に節約できる。

コンテナを起動すると、イメージの最上部に書き込み可能なレイヤー（コンテナレイヤー）が追加される。コンテナ内での変更はすべてこのレイヤーに記録され、コンテナを削除するとこのレイヤーも削除される。これが「コンテナはエフェメラル（一時的）」と言われる理由である。

### 1.2 レイヤーの確認

```bash
# イメージのレイヤー情報を表示
docker history nginx:alpine

# 出力例:
# IMAGE          CREATED       CREATED BY                                      SIZE
# a1b2c3d4       2 days ago    CMD ["nginx" "-g" "daemon off;"]                0B
# <missing>      2 days ago    EXPOSE map[80/tcp:{}]                           0B
# <missing>      2 days ago    COPY docker-entrypoint.sh /                     4.62kB
# <missing>      2 days ago    RUN /bin/sh -c set -x && addgroup...            26.7MB
# <missing>      2 days ago    ENV NGINX_VERSION=1.25.3                        0B
# <missing>      3 weeks ago   /bin/sh -c #(nop) CMD ["/bin/sh"]               0B
# <missing>      3 weeks ago   ADD file:xxx in /                               7.38MB

# サイズなしでコマンドのみ表示
docker history --no-trunc --format "{{.CreatedBy}}" nginx:alpine

# イメージの詳細情報
docker inspect nginx:alpine

# JSON 形式で特定のフィールドを抽出
docker inspect --format '{{json .RootFS.Layers}}' nginx:alpine | python3 -m json.tool

# レイヤー数のカウント
docker inspect --format '{{len .RootFS.Layers}}' nginx:alpine
```

### 1.3 レイヤーの共有と効率性

```bash
# 同じベースイメージを使う2つのイメージのディスク使用量を確認
docker pull node:20-alpine
docker pull node:18-alpine

# 各イメージのサイズ
docker images node
# REPOSITORY   TAG         IMAGE ID       CREATED       SIZE
# node         20-alpine   abc123         2 days ago    130MB
# node         18-alpine   def456         5 days ago    125MB

# 実際のディスク使用量（レイヤー共有を考慮）
docker system df -v
# -> Shared Size が表示される

# レイヤーの共有状況を確認
docker inspect --format '{{.RootFS.Layers}}' node:20-alpine
docker inspect --format '{{.RootFS.Layers}}' node:18-alpine
# -> 共通するレイヤーハッシュがあれば、そのレイヤーは共有されている
```

### 1.4 Copy-on-Write (CoW) の仕組み

```
+------------------------------------------------------+
|           Copy-on-Write の動作                        |
|                                                      |
|  コンテナ起動時:                                      |
|  +--------------------------------------------+      |
|  | コンテナレイヤー (R/W) - 空               |      |
|  +--------------------------------------------+      |
|  | Layer 3 (R/O) - /app/server.js            |      |
|  +--------------------------------------------+      |
|  | Layer 2 (R/O) - /usr/lib/...              |      |
|  +--------------------------------------------+      |
|  | Layer 1 (R/O) - ベースOS                   |      |
|  +--------------------------------------------+      |
|                                                      |
|  ファイル読み取り: 上から下に検索、最初に見つかった     |
|  レイヤーのファイルを返す                               |
|                                                      |
|  ファイル書き込み: 対象ファイルをコンテナレイヤーに      |
|  コピーしてから変更（Copy-on-Write）                   |
|                                                      |
|  ファイル削除: whiteout ファイルでマスク               |
|  （下のレイヤーのファイルは実際には削除されない）         |
+------------------------------------------------------+
```

---

## 2. イメージの取得 (pull)

### 2.1 基本操作

```bash
# 最新バージョンを取得
docker pull nginx
# -> nginx:latest が取得される

# 特定バージョンを指定
docker pull nginx:1.25.3-alpine

# 特定のプラットフォームを指定
docker pull --platform linux/arm64 nginx:alpine

# ダイジェスト（SHA256）で指定（完全な再現性）
docker pull nginx@sha256:abc123def456...

# 複数のタグを一度に取得
docker pull nginx:1.25.3-alpine
docker pull nginx:1.25.3-bookworm

# 全タグをリストする（Docker Hub API）
curl -s "https://registry.hub.docker.com/v2/repositories/library/nginx/tags?page_size=100" | \
  python3 -c "import sys,json;[print(t['name']) for t in json.load(sys.stdin)['results']]"
```

### 2.2 レジストリからの取得

```bash
# Docker Hub（デフォルト）
docker pull nginx:alpine
# -> docker.io/library/nginx:alpine と同じ

# GitHub Container Registry
docker pull ghcr.io/owner/image:tag

# Amazon ECR
docker pull 123456789.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:v1

# Google Artifact Registry
docker pull asia-northeast1-docker.pkg.dev/project/repo/image:tag

# Azure Container Registry
docker pull myregistry.azurecr.io/my-app:v1

# セルフホストレジストリ
docker pull registry.example.com:5000/my-app:v1

# Red Hat Quay
docker pull quay.io/organization/image:tag
```

### 2.3 pull の高速化と効率化

```bash
# 並列ダウンロードの設定（daemon.json）
# /etc/docker/daemon.json
# {
#   "max-concurrent-downloads": 10,
#   "max-concurrent-uploads": 5
# }

# ミラーレジストリの設定（Rate Limit 対策）
# /etc/docker/daemon.json
# {
#   "registry-mirrors": [
#     "https://mirror.gcr.io",
#     "https://docker-mirror.example.com"
#   ]
# }

# プルポリシーの確認（Kubernetes / Docker Compose）
# imagePullPolicy: IfNotPresent  -> ローカルになければ pull
# imagePullPolicy: Always        -> 常に pull
# imagePullPolicy: Never         -> ローカルのみ使用

# pull の進捗を確認
docker pull --quiet nginx:alpine  # 進捗を非表示にする
docker pull nginx:alpine 2>&1 | tail -1  # 最終結果のみ
```

---

## 3. タグ体系

### 3.1 タグの命名規則

```
+------------------------------------------------------+
|                  イメージタグの構造                     |
|                                                      |
|  レジストリ / 名前空間 / リポジトリ : タグ              |
|                                                      |
|  例:                                                 |
|  docker.io / library   / nginx     : 1.25.3-alpine   |
|  ghcr.io   / myorg     / myapp     : v2.1.0          |
|  (省略可)    (省略可)                 (デフォルト:latest)|
|                                                      |
|  タグ命名のベストプラクティス:                          |
|  +------------------------------------------------+ |
|  | セマンティックバージョニング: v1.2.3              | |
|  | Git SHA: sha-abc123f                            | |
|  | 日付: 2024-01-15                                | |
|  | 環境: production, staging                       | |
|  | ベース指定: 1.25.3-alpine, 1.25.3-bookworm      | |
|  +------------------------------------------------+ |
+------------------------------------------------------+
```

```bash
# タグの付与
docker tag my-app:latest my-app:v1.0.0
docker tag my-app:latest my-app:v1.0
docker tag my-app:latest my-app:v1

# リモートレジストリ用のタグ付け
docker tag my-app:v1.0.0 ghcr.io/myorg/my-app:v1.0.0
docker tag my-app:v1.0.0 ghcr.io/myorg/my-app:latest

# タグの一覧
docker images my-app
# REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
# my-app       latest    abc123def456   1 hour ago    150MB
# my-app       v1.0.0    abc123def456   1 hour ago    150MB
# my-app       v1.0      abc123def456   1 hour ago    150MB
# my-app       v1        abc123def456   1 hour ago    150MB
# (同じ IMAGE ID = 同じイメージに複数のタグ)
```

### 3.2 セマンティックバージョニングの実践

```bash
# CI/CD でのタグ戦略の実装例
#!/bin/bash
# build-and-tag.sh

APP_NAME="my-app"
REGISTRY="ghcr.io/myorg"
VERSION=$(cat version.txt)              # 例: 1.2.3
GIT_SHA=$(git rev-parse --short HEAD)   # 例: abc123f
BUILD_DATE=$(date -u +%Y%m%d)          # 例: 20240115
BRANCH=$(git branch --show-current)     # 例: main

# ビルド
docker build \
  --label "org.opencontainers.image.version=${VERSION}" \
  --label "org.opencontainers.image.revision=${GIT_SHA}" \
  --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t "${REGISTRY}/${APP_NAME}:${VERSION}" \
  -t "${REGISTRY}/${APP_NAME}:${VERSION}-${GIT_SHA}" \
  -t "${REGISTRY}/${APP_NAME}:sha-${GIT_SHA}" \
  .

# main ブランチなら latest もタグ付け
if [ "$BRANCH" = "main" ]; then
  docker tag "${REGISTRY}/${APP_NAME}:${VERSION}" "${REGISTRY}/${APP_NAME}:latest"
fi

# プッシュ
docker push "${REGISTRY}/${APP_NAME}" --all-tags
```

### 3.3 OCI イメージラベルの標準

```dockerfile
# Dockerfile 内でのラベル設定（OCI Image Spec 準拠）
LABEL org.opencontainers.image.title="My Application"
LABEL org.opencontainers.image.description="A web application"
LABEL org.opencontainers.image.version="1.2.3"
LABEL org.opencontainers.image.authors="team@example.com"
LABEL org.opencontainers.image.url="https://github.com/myorg/my-app"
LABEL org.opencontainers.image.source="https://github.com/myorg/my-app"
LABEL org.opencontainers.image.documentation="https://docs.example.com"
LABEL org.opencontainers.image.vendor="My Organization"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.created="2024-01-15T10:00:00Z"
LABEL org.opencontainers.image.revision="abc123f"
```

### 比較表 1: タグ戦略

| 戦略 | 例 | 用途 | メリット | デメリット |
|---|---|---|---|---|
| セマンティック | `v1.2.3` | リリース管理 | バージョンが明確 | 手動管理が必要 |
| Git SHA | `sha-abc123f` | CI/CD | 完全なトレーサビリティ | 人間に分かりにくい |
| 日付 | `2024-01-15` | 定期ビルド | 時系列が明確 | 1日複数ビルドで衝突 |
| latest | `latest` | 開発テスト | 常に最新 | 再現性なし |
| 環境 | `production` | デプロイ管理 | 直感的 | ミュータブル |
| ブランチ名 | `feature-auth` | PR/開発 | 追跡が容易 | ブランチ削除後に孤児化 |
| 複合 | `v1.2.3-sha-abc123f` | 本番リリース | 完全な識別 | タグが長い |

---

## 4. イメージの配布 (push)

### 4.1 Docker Hub

```bash
# Docker Hub にログイン
docker login
# Username: myuser
# Password: ****

# イメージにタグを付ける
docker tag my-app:v1.0.0 myuser/my-app:v1.0.0

# プッシュ
docker push myuser/my-app:v1.0.0

# 全タグをプッシュ
docker push myuser/my-app --all-tags

# ログアウト
docker logout
```

### 4.2 GitHub Container Registry (GHCR)

```bash
# GitHub Personal Access Token でログイン
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# タグ付けとプッシュ
docker tag my-app:v1.0.0 ghcr.io/myorg/my-app:v1.0.0
docker push ghcr.io/myorg/my-app:v1.0.0
```

```yaml
# GitHub Actions からのプッシュ
# .github/workflows/publish.yml
name: Build and Push
on:
  push:
    tags: ['v*']

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

### 4.3 Amazon ECR

```bash
# AWS CLI で ECR にログイン
aws ecr get-login-password --region ap-northeast-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.ap-northeast-1.amazonaws.com

# リポジトリの作成（初回のみ）
aws ecr create-repository --repository-name my-app \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration encryptionType=AES256

# タグ付けとプッシュ
docker tag my-app:v1.0.0 123456789.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:v1.0.0
docker push 123456789.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:v1.0.0

# ライフサイクルポリシーの設定（古いイメージの自動削除）
aws ecr put-lifecycle-policy --repository-name my-app --lifecycle-policy-text '{
  "rules": [
    {
      "rulePriority": 1,
      "description": "Keep last 10 images",
      "selection": {
        "tagStatus": "any",
        "countType": "imageCountMoreThan",
        "countNumber": 10
      },
      "action": {
        "type": "expire"
      }
    }
  ]
}'
```

### 4.4 Google Artifact Registry

```bash
# gcloud CLI で認証
gcloud auth configure-docker asia-northeast1-docker.pkg.dev

# リポジトリの作成（初回のみ）
gcloud artifacts repositories create my-repo \
  --repository-format=docker \
  --location=asia-northeast1 \
  --description="My Docker repository"

# タグ付けとプッシュ
docker tag my-app:v1.0.0 asia-northeast1-docker.pkg.dev/my-project/my-repo/my-app:v1.0.0
docker push asia-northeast1-docker.pkg.dev/my-project/my-repo/my-app:v1.0.0

# イメージ一覧
gcloud artifacts docker images list asia-northeast1-docker.pkg.dev/my-project/my-repo
```

### 4.5 プライベートレジストリの構築

```bash
# Docker Registry をローカルで起動
docker run -d -p 5000:5000 --name registry registry:2

# ローカルレジストリにプッシュ
docker tag my-app:v1.0.0 localhost:5000/my-app:v1.0.0
docker push localhost:5000/my-app:v1.0.0

# ローカルレジストリからプル
docker pull localhost:5000/my-app:v1.0.0

# レジストリのカタログ確認
curl http://localhost:5000/v2/_catalog
# {"repositories":["my-app"]}

# タグ一覧の確認
curl http://localhost:5000/v2/my-app/tags/list
# {"name":"my-app","tags":["v1.0.0"]}
```

```yaml
# docker-compose.yml による本格的なプライベートレジストリ
services:
  registry:
    image: registry:2
    ports:
      - "5000:5000"
    volumes:
      - registry_data:/var/lib/registry
      - ./certs:/certs
      - ./auth:/auth
    environment:
      REGISTRY_HTTP_TLS_CERTIFICATE: /certs/domain.crt
      REGISTRY_HTTP_TLS_KEY: /certs/domain.key
      REGISTRY_AUTH: htpasswd
      REGISTRY_AUTH_HTPASSWD_PATH: /auth/htpasswd
      REGISTRY_AUTH_HTPASSWD_REALM: "Registry Realm"
      REGISTRY_STORAGE_DELETE_ENABLED: "true"
    restart: always

  registry-ui:
    image: joxit/docker-registry-ui:latest
    ports:
      - "8080:80"
    environment:
      REGISTRY_URL: https://registry:5000
      DELETE_IMAGES: "true"
      SINGLE_REGISTRY: "true"
    depends_on:
      - registry

volumes:
  registry_data:
```

```bash
# htpasswd ファイルの作成
docker run --rm --entrypoint htpasswd registry:2 -Bbn myuser mypassword > auth/htpasswd

# TLS 証明書の作成（自己署名、開発用）
openssl req -newkey rsa:4096 -nodes -sha256 -keyout certs/domain.key \
  -x509 -days 365 -out certs/domain.crt \
  -subj "/CN=registry.example.com"
```

---

## 5. イメージの検査

### 5.1 イメージ情報の確認

```bash
# 基本情報
docker images
# REPOSITORY   TAG            IMAGE ID       CREATED        SIZE
# nginx        alpine         a1b2c3d4       2 days ago     42.6MB
# node         20-alpine      e5f6g7h8       1 week ago     181MB
# postgres     16-alpine      i9j0k1l2       3 days ago     244MB

# 詳細情報
docker inspect nginx:alpine

# 特定の情報を抽出
docker inspect --format '{{.Config.Env}}' nginx:alpine
docker inspect --format '{{.Config.ExposedPorts}}' nginx:alpine
docker inspect --format '{{.Config.Cmd}}' nginx:alpine
docker inspect --format '{{.Architecture}}' nginx:alpine
docker inspect --format '{{.Os}}' nginx:alpine

# ラベル情報の取得
docker inspect --format '{{json .Config.Labels}}' nginx:alpine | python3 -m json.tool

# イメージのエントリポイントを確認
docker inspect --format '{{json .Config.Entrypoint}}' nginx:alpine

# マニフェストの確認（マルチプラットフォーム対応確認）
docker manifest inspect nginx:alpine

# イメージサイズの詳細（圧縮前後）
docker manifest inspect --verbose nginx:alpine | \
  python3 -c "import sys,json;d=json.load(sys.stdin);print(f'Compressed: {sum(l[\"size\"] for l in d[\"SchemaV2Manifest\"][\"layers\"])/1e6:.1f}MB')"
```

### 5.2 イメージの内容を探索

```bash
# イメージからコンテナを作成して中身を確認
docker run --rm -it nginx:alpine /bin/sh

# イメージをtarにエクスポート
docker save nginx:alpine -o nginx-alpine.tar
# tarの中身を確認
tar tf nginx-alpine.tar | head -20

# 特定のファイルだけ確認
docker run --rm nginx:alpine cat /etc/nginx/nginx.conf

# ファイルシステムの差分を確認（コンテナが変更したファイル）
docker diff my-running-container
# A /tmp/newfile    (Added)
# C /var/log        (Changed)
# D /tmp/oldfile    (Deleted)

# イメージ内のファイル一覧を取得
docker run --rm nginx:alpine find / -type f 2>/dev/null | head -50

# 特定のパッケージがインストールされているか確認
docker run --rm nginx:alpine apk list --installed 2>/dev/null
docker run --rm python:3.12-slim dpkg -l 2>/dev/null | head -30
```

### 5.3 dive によるイメージ分析

```bash
# dive のインストール
brew install dive  # macOS
# または Docker で実行
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  wagoodman/dive nginx:alpine

# dive でイメージを分析
dive nginx:alpine

# CI モードでイメージ効率をチェック
dive --ci nginx:alpine
# -> イメージ効率スコアが表示される
# -> 無駄なレイヤーや重複ファイルが検出される
```

### 5.4 SBOM（Software Bill of Materials）の生成

```bash
# Docker SBOM（Docker Desktop 統合）
docker sbom nginx:alpine

# Syft による SBOM 生成
# インストール
brew install syft  # macOS

# SBOM の生成（SPDX 形式）
syft nginx:alpine -o spdx-json > sbom.spdx.json

# SBOM の生成（CycloneDX 形式）
syft nginx:alpine -o cyclonedx-json > sbom.cdx.json

# テキスト形式で表示
syft nginx:alpine

# SBOM に基づく脆弱性スキャン
syft nginx:alpine -o spdx-json | grype
```

---

## 6. レジストリ比較

### 比較表 2: コンテナレジストリ比較

| レジストリ | 無料枠 | プライベートリポジトリ | 主な用途 | 認証方法 |
|---|---|---|---|---|
| Docker Hub | 1プライベートリポ | 有料プラン | OSS配布、公式イメージ | Docker ID |
| GitHub Container Registry | 無制限（パブリック） | GitHub プランに依存 | GitHub統合プロジェクト | GitHub PAT |
| Amazon ECR | 500MB/月(パブリック) | AWS料金 | AWS本番環境 | IAM |
| Google Artifact Registry | 500MB/月 | GCP料金 | GCP本番環境 | gcloud auth |
| Azure Container Registry | なし | Azure料金 | Azure本番環境 | Azure AD |
| Harbor (OSS) | セルフホスト | 無制限 | オンプレミス | LDAP/OIDC |
| Quay.io (Red Hat) | 無制限（パブリック） | 有料プラン | Red Hat エコシステム | Red Hat SSO |
| GitLab Container Registry | GitLab プランに依存 | GitLab プランに依存 | GitLab CI/CD 統合 | GitLab トークン |

### 比較表 3: コスト比較（月額目安）

| レジストリ | 小規模 (10GB) | 中規模 (100GB) | 大規模 (1TB) |
|---|---|---|---|
| Docker Hub (Team) | $7/user | $7/user | $7/user (ストレージ無制限) |
| GHCR (GitHub Team) | $4/user | $4/user + ストレージ | $4/user + ストレージ |
| Amazon ECR | ~$1 | ~$10 | ~$100 |
| Google AR | ~$2.6 | ~$26 | ~$260 |
| Azure ACR (Basic) | $5 (10GB含) | $50 (100GB含) | $200+ |
| Harbor | サーバー費用のみ | サーバー費用のみ | サーバー費用のみ |

---

## 7. イメージのセキュリティ

### 7.1 脆弱性スキャン

```
+------------------------------------------------------+
|              イメージセキュリティスキャン                |
|                                                      |
|  イメージ                                             |
|    |                                                 |
|    v                                                 |
|  +------------------+                                |
|  | 脆弱性スキャナー   |                                |
|  +-----|------------+                                |
|        |                                             |
|  +-----+------+--------+--------+                    |
|  |     |      |        |        |                    |
|  v     v      v        v        v                    |
| Docker Trivy  Snyk   Grype   Clair                  |
| Scout                                               |
|                                                      |
|  スキャン対象:                                        |
|  - OS パッケージ (apt/apk/yum)                       |
|  - 言語パッケージ (npm/pip/go mod)                    |
|  - 設定ファイルの問題                                  |
|  - シークレットの検出                                  |
|  - ライセンスコンプライアンス                            |
+------------------------------------------------------+
```

```bash
# Docker Scout（Docker Desktop 統合）
docker scout quickview nginx:alpine
docker scout cves nginx:alpine
docker scout recommendations nginx:alpine

# Trivy（オープンソース、推奨）
# インストール
brew install aquasecurity/trivy/trivy  # macOS
# または Docker で実行
docker run --rm aquasec/trivy image nginx:alpine

# Trivy でイメージスキャン
trivy image nginx:alpine

# 重要度フィルタ
trivy image --severity HIGH,CRITICAL nginx:alpine

# 修正可能な脆弱性のみ表示
trivy image --ignore-unfixed nginx:alpine

# JSON 形式で出力（CI/CD での利用）
trivy image --format json --output result.json nginx:alpine

# テーブル形式で出力
trivy image --format table nginx:alpine

# 終了コードで CI を制御（CRITICAL があれば失敗）
trivy image --exit-code 1 --severity CRITICAL nginx:alpine

# Grype
docker run --rm anchore/grype nginx:alpine

# Grype で特定の脆弱性を無視
echo "CVE-2023-12345" > .grype.yaml
grype nginx:alpine --config .grype.yaml
```

### 7.2 イメージ署名

```bash
# Docker Content Trust (DCT) を有効化
export DOCKER_CONTENT_TRUST=1

# 署名付きでプッシュ
docker push myuser/my-app:v1.0.0
# 初回は署名鍵のパスフレーズ設定が求められる

# 署名されたイメージのみプル可能
docker pull myuser/my-app:v1.0.0

# cosign による署名（Sigstore）
# インストール
brew install cosign  # macOS

# 鍵ペアの生成
cosign generate-key-pair

# イメージの署名
cosign sign --key cosign.key ghcr.io/myorg/my-app:v1.0.0

# 署名の検証
cosign verify --key cosign.pub ghcr.io/myorg/my-app:v1.0.0

# キーレス署名（GitHub Actions OIDC 連携）
cosign sign ghcr.io/myorg/my-app:v1.0.0
# -> Sigstore の透明性ログ (Rekor) に記録される

# 署名の確認
cosign verify \
  --certificate-identity "https://github.com/myorg/my-app/.github/workflows/build.yml@refs/tags/v1.0.0" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  ghcr.io/myorg/my-app:v1.0.0
```

### 7.3 セキュリティベストプラクティス

```
+------------------------------------------------------+
|       イメージセキュリティのベストプラクティス            |
|                                                      |
|  1. 最小ベースイメージの使用                            |
|     Alpine > Debian Slim > Ubuntu                    |
|     Distroless > Alpine (攻撃面の最小化)              |
|                                                      |
|  2. non-root ユーザーでの実行                          |
|     USER appuser (root 権限を回避)                    |
|                                                      |
|  3. 読み取り専用ファイルシステム                        |
|     docker run --read-only --tmpfs /tmp ...           |
|                                                      |
|  4. 定期的な脆弱性スキャン                              |
|     CI/CD パイプラインに Trivy を統合                  |
|                                                      |
|  5. イメージの署名と検証                                |
|     cosign + Sigstore で供給チェーンを保護             |
|                                                      |
|  6. シークレットをイメージに含めない                     |
|     ARG/ENV ではなく --mount=type=secret を使用       |
|                                                      |
|  7. .dockerignore の徹底                              |
|     .env, .git, credentials を除外                   |
+------------------------------------------------------+
```

---

## 8. マルチプラットフォームビルド

### 8.1 buildx によるマルチプラットフォーム対応

```bash
# buildx ビルダーの確認
docker buildx ls

# マルチプラットフォーム用ビルダーの作成
docker buildx create --name multiplatform --driver docker-container --use

# マルチプラットフォームビルド & プッシュ
docker buildx build \
  --platform linux/amd64,linux/arm64,linux/arm/v7 \
  -t ghcr.io/myorg/my-app:v1.0.0 \
  --push \
  .

# 特定のプラットフォームのみビルド
docker buildx build \
  --platform linux/amd64 \
  -t my-app:v1.0.0 \
  --load \
  .

# ビルド済みイメージの対応プラットフォーム確認
docker manifest inspect ghcr.io/myorg/my-app:v1.0.0

# QEMU エミュレーションのセットアップ（異なるアーキテクチャのビルド用）
docker run --privileged --rm tonistiigi/binfmt --install all
```

### 8.2 マルチプラットフォーム対応 Dockerfile

```dockerfile
# マルチプラットフォーム対応の Dockerfile 例
FROM --platform=$BUILDPLATFORM golang:1.22-alpine AS builder

# ビルドプラットフォーム情報
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETOS
ARG TARGETARCH

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
# クロスコンパイル
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -ldflags="-w -s" -o /server ./cmd/server

FROM scratch
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
```

```bash
# GitHub Actions でのマルチプラットフォームビルド
# .github/workflows/multi-platform.yml
```

```yaml
name: Multi-platform Build
on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-qemu-action@v3

      - uses: docker/setup-buildx-action@v3

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## 9. イメージのクリーンアップ

```bash
# ローカルイメージの一覧
docker images

# 特定のイメージを削除
docker rmi nginx:alpine

# 強制削除（使用中のコンテナがあっても削除）
docker rmi -f nginx:alpine

# dangling イメージ（タグなし）を削除
docker image prune

# 未使用の全イメージを削除
docker image prune -a

# 特定の条件でフィルタリングして削除
docker image prune -a --filter "until=24h"  # 24時間以上前のイメージ
docker image prune -a --filter "label=env=dev"  # ラベル指定
docker image prune -a --filter "label!=keep=true"  # 保持ラベルがないもの

# 特定のパターンに一致するイメージを一括削除
docker images --format '{{.Repository}}:{{.Tag}}' | grep 'my-app' | xargs docker rmi

# dangling イメージのみ一覧表示
docker images --filter "dangling=true"

# ディスク使用量の確認
docker system df
# TYPE            TOTAL   ACTIVE   SIZE      RECLAIMABLE
# Images          25      5        8.5GB     6.2GB (72%)
# Containers      10      3        250MB     180MB (72%)
# Local Volumes   8       4        2.1GB     900MB (42%)
# Build Cache     100     0        3.5GB     3.5GB (100%)

# 詳細なディスク使用量
docker system df -v

# ビルドキャッシュのクリーンアップ
docker builder prune
docker builder prune --all  # 全てのビルドキャッシュを削除
docker builder prune --keep-storage 5GB  # 5GB を超える分のみ削除

# 完全クリーンアップ
docker system prune -a --volumes
```

### クリーンアップの自動化

```bash
#!/bin/bash
# docker-cleanup.sh - 定期実行用クリーンアップスクリプト

echo "=== Docker クリーンアップ開始 ==="

# 停止中のコンテナを削除
echo "--- 停止中のコンテナを削除 ---"
docker container prune -f

# dangling イメージを削除
echo "--- dangling イメージを削除 ---"
docker image prune -f

# 7日以上前の未使用イメージを削除
echo "--- 7日以上前の未使用イメージを削除 ---"
docker image prune -a -f --filter "until=168h"

# 未使用のボリュームを削除
echo "--- 未使用のボリュームを削除 ---"
docker volume prune -f

# 未使用のネットワークを削除
echo "--- 未使用のネットワークを削除 ---"
docker network prune -f

# ビルドキャッシュを 10GB 以内に制限
echo "--- ビルドキャッシュのクリーンアップ ---"
docker builder prune -f --keep-storage 10GB

# 最終的なディスク使用量を表示
echo "=== クリーンアップ完了 ==="
docker system df
```

```bash
# cron で毎日深夜に実行
# crontab -e
0 3 * * * /usr/local/bin/docker-cleanup.sh >> /var/log/docker-cleanup.log 2>&1
```

---

## 10. イメージの保存と転送

```bash
# イメージをファイルに保存
docker save -o my-app-v1.tar my-app:v1.0.0

# 圧縮して保存
docker save my-app:v1.0.0 | gzip > my-app-v1.tar.gz

# zstd 圧縮（より高速・高圧縮）
docker save my-app:v1.0.0 | zstd > my-app-v1.tar.zst

# ファイルからイメージを読み込み
docker load -i my-app-v1.tar

# 圧縮ファイルから読み込み
gunzip -c my-app-v1.tar.gz | docker load
zstd -d -c my-app-v1.tar.zst | docker load

# 複数イメージを1つの tar にまとめる
docker save -o all-images.tar my-app:v1.0.0 nginx:alpine postgres:16-alpine

# コンテナの現在の状態をイメージとして保存
docker commit my-container my-app:snapshot
# 注意: commit は開発では非推奨。Dockerfile を使うべき。

# SSH 経由で他のホストにイメージを転送
docker save my-app:v1.0.0 | gzip | ssh user@remote "gunzip | docker load"

# コンテナのファイルシステムを tar としてエクスポート（レイヤー情報は失われる）
docker export my-container > container-fs.tar
docker import container-fs.tar my-app:imported
```

### 比較表 4: イメージ転送方法

| 方法 | 速度 | 用途 | メリット | デメリット |
|---|---|---|---|---|
| レジストリ (push/pull) | 高速 | 通常の開発・デプロイ | レイヤーキャッシュが効く | レジストリが必要 |
| docker save/load | 中速 | オフライン環境 | レジストリ不要 | 全レイヤーを含む |
| docker export/import | 低速 | ファイルシステム抽出 | フラットな tar | レイヤー情報消失 |
| SSH 直接転送 | 中速 | 緊急時 | インフラ不要 | 帯域に依存 |
| 外部ストレージ | 中速 | CI/CD キャッシュ | S3 等と統合 | 設定が必要 |

---

## 11. CI/CD でのイメージ管理

### 11.1 GitHub Actions でのビルドとプッシュ

```yaml
# .github/workflows/docker.yml
name: Docker Build

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
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      security-events: write

    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-buildx-action@v3

      - uses: docker/login-action@v3
        if: github.event_name != 'pull_request'
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      # 脆弱性スキャン
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.meta.outputs.version }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'
```

### 11.2 GitLab CI でのビルド

```yaml
# .gitlab-ci.yml
stages:
  - build
  - scan
  - push

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE

build:
  stage: build
  image: docker:24-dind
  services:
    - docker:24-dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $DOCKER_IMAGE:$CI_COMMIT_SHA .
    - docker push $DOCKER_IMAGE:$CI_COMMIT_SHA

scan:
  stage: scan
  image:
    name: aquasec/trivy
    entrypoint: [""]
  script:
    - trivy image --exit-code 1 --severity CRITICAL $DOCKER_IMAGE:$CI_COMMIT_SHA

push:
  stage: push
  image: docker:24-dind
  services:
    - docker:24-dind
  only:
    - tags
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker pull $DOCKER_IMAGE:$CI_COMMIT_SHA
    - docker tag $DOCKER_IMAGE:$CI_COMMIT_SHA $DOCKER_IMAGE:$CI_COMMIT_TAG
    - docker tag $DOCKER_IMAGE:$CI_COMMIT_SHA $DOCKER_IMAGE:latest
    - docker push $DOCKER_IMAGE:$CI_COMMIT_TAG
    - docker push $DOCKER_IMAGE:latest
```

---

## 12. アンチパターン

### アンチパターン 1: latest タグだけで管理する

```bash
# NG: 全て latest でプッシュ
docker build -t my-app .
docker push my-app:latest
# -> どのバージョンがデプロイされているか分からない
# -> ロールバックできない
# -> キャッシュの挙動が予測できない

# OK: セマンティックバージョニング + Git SHA
docker build -t my-app:v1.2.3 -t my-app:sha-abc123f .
docker push my-app:v1.2.3
docker push my-app:sha-abc123f
# -> バージョン特定可能、ロールバック容易
```

### アンチパターン 2: docker commit でイメージを作成する

```bash
# NG: コンテナに手動変更してcommit
docker run -it ubuntu bash
# (コンテナ内で apt install, ファイル編集等)
docker commit <container-id> my-custom-image
# -> 再現性がない
# -> 変更内容が不明
# -> レビューできない

# OK: Dockerfile でイメージを定義
# Dockerfile
# FROM ubuntu:22.04
# RUN apt-get update && apt-get install -y curl
# COPY config.json /etc/app/
docker build -t my-custom-image .
# -> 再現性あり、レビュー可能、バージョン管理可能
```

### アンチパターン 3: ビルド時にシークレットを ARG/ENV で渡す

```dockerfile
# NG: シークレットが レイヤーに残る
FROM node:20-alpine
WORKDIR /app
ARG NPM_TOKEN
COPY .npmrc .
RUN npm ci
RUN rm .npmrc  # 削除しても前のレイヤーに残っている！

# OK: BuildKit のシークレットマウントを使う
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc \
    npm ci
COPY . .
CMD ["node", "server.js"]
```

```bash
# ビルド時にシークレットを渡す
docker build --secret id=npmrc,src=.npmrc -t my-app .
```

### アンチパターン 4: 巨大なベースイメージを使う

```dockerfile
# NG: フルサイズの ubuntu を使用
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y nodejs npm
COPY . /app
CMD ["node", "/app/server.js"]
# -> 400MB+ のイメージ

# OK: 言語固有の Alpine イメージを使用
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --only=production
COPY . .
CMD ["node", "server.js"]
# -> 150MB 程度のイメージ
```

---

## 13. FAQ

### Q1: `docker pull` は毎回イメージ全体をダウンロードするのですか？

**A:** いいえ。Docker はレイヤー単位でダウンロードし、ローカルに既存のレイヤーはスキップする。そのため、ベースイメージが同じ場合、2 回目以降のプルは差分のみで非常に高速になる。`docker pull` の出力で `Already exists` と表示されるレイヤーはキャッシュが使われている。

### Q2: Docker Hub の Rate Limit はどの程度ですか？

**A:** 匿名ユーザーは 6 時間あたり 100 プル、無料アカウントは 6 時間あたり 200 プルの制限がある。CI/CD で頻繁にプルする場合は、Docker Hub の有料プランか、GitHub Container Registry 等の代替レジストリの利用を検討すべきである。ミラーレジストリを構築して Rate Limit を回避する方法もある。Rate Limit の状況は以下のコマンドで確認できる:

```bash
# Rate Limit の残り回数を確認
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/nginx:pull" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -sI -H "Authorization: Bearer $TOKEN" https://registry-1.docker.io/v2/library/nginx/manifests/latest 2>&1 | grep -i ratelimit
```

### Q3: マルチプラットフォームイメージとは何ですか？

**A:** 1 つのタグで複数のアーキテクチャ（amd64, arm64 等）に対応するイメージのことである。`docker manifest` で管理され、`docker pull` 時にホストのアーキテクチャに合ったイメージが自動的に選択される。`docker buildx build --platform linux/amd64,linux/arm64` で作成できる。Apple Silicon Mac (arm64) と Linux サーバー (amd64) の両方で動作するイメージを1つのタグで管理したい場合に特に有用である。

### Q4: イメージサイズを確認する方法は？

**A:** `docker images` でローカルのサイズを、`docker manifest inspect` でレジストリ上のサイズを確認できる。ただし、`docker images` の SIZE はレイヤー共有を考慮しない見かけのサイズである。実際のディスク使用量は `docker system df -v` で確認する。イメージのレイヤーごとのサイズは `docker history` で確認できる。さらに詳細な分析には `dive` ツールが有効で、各レイヤーの内容とイメージ効率スコアを可視化できる。

### Q5: イメージの脆弱性が見つかった場合、どう対応すべきですか？

**A:** 対応の優先度は以下の通り:
1. **CRITICAL**: 即座に対応。ベースイメージの更新、パッケージの更新で修正。
2. **HIGH**: 次のリリースまでに対応。
3. **MEDIUM/LOW**: 計画的に対応。修正パッケージがリリースされるまで待機する場合もある。
ベースイメージを定期的に再ビルドすることで、OS パッケージの脆弱性は自動的に修正されることが多い。`--ignore-unfixed` オプションで修正が提供されていない脆弱性を除外してスキャンすることも有効である。

### Q6: プライベートレジストリのバックアップはどうすべきですか？

**A:** 以下の方法がある:
- **ボリュームバックアップ**: レジストリのデータボリューム (`/var/lib/registry`) をバックアップする
- **S3 バックエンド**: レジストリのストレージを S3 に設定し、S3 のバージョニングとレプリケーションでバックアップ
- **レジストリ間ミラーリング**: `skopeo` ツールを使って別のレジストリにイメージをコピー
- **定期 save**: 重要なイメージを `docker save` で定期的にファイルに保存

```bash
# skopeo でのイメージコピー
skopeo copy \
  docker://registry.example.com/my-app:v1.0.0 \
  docker://backup-registry.example.com/my-app:v1.0.0

# 全イメージの一括バックアップスクリプト
for repo in $(curl -s http://registry:5000/v2/_catalog | python3 -c "import sys,json;[print(r) for r in json.load(sys.stdin)['repositories']]"); do
  for tag in $(curl -s "http://registry:5000/v2/${repo}/tags/list" | python3 -c "import sys,json;[print(t) for t in json.load(sys.stdin)['tags']]"); do
    skopeo copy "docker://registry:5000/${repo}:${tag}" "dir:/backup/${repo}/${tag}"
  done
done
```

---

## 14. まとめ

| 項目 | ポイント |
|---|---|
| レイヤー構造 | イメージはレイヤーの積み重ね。変更されないレイヤーはキャッシュ再利用 |
| タグ | セマンティックバージョニング推奨。latest 依存は避ける |
| レジストリ | Docker Hub, GHCR, ECR 等。用途とコストで選択 |
| セキュリティ | Trivy / Docker Scout で定期的に脆弱性スキャン |
| マルチプラットフォーム | buildx で amd64/arm64 両対応のイメージを構築 |
| 署名 | cosign + Sigstore でイメージの真正性を保証 |
| SBOM | Syft で SBOM を生成し、依存関係を可視化 |
| クリーンアップ | `docker system prune` で定期的にディスクを解放 |
| 保存・転送 | `docker save/load` でオフライン環境にも対応 |
| CI/CD | GitHub Actions / GitLab CI でビルド・スキャン・プッシュを自動化 |
| ベストプラクティス | Dockerfile で管理、commit は非推奨、最小ベースイメージ使用 |

---

## 次に読むべきガイド

- [../01-dockerfile/00-dockerfile-basics.md](../01-dockerfile/00-dockerfile-basics.md) -- Dockerfile の基礎
- [../01-dockerfile/01-multi-stage-build.md](../01-dockerfile/01-multi-stage-build.md) -- マルチステージビルド
- [../01-dockerfile/02-optimization.md](../01-dockerfile/02-optimization.md) -- Dockerfile の最適化

---

## 参考文献

1. **Docker Documentation - Docker Hub** https://docs.docker.com/docker-hub/ -- Docker Hub の公式ドキュメント。リポジトリ管理、組織設定、Rate Limit の詳細。
2. **GitHub Documentation - Working with the Container registry** https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry -- GitHub Container Registry の利用方法。
3. **Aqua Security - Trivy** https://aquasecurity.github.io/trivy/ -- Trivy の公式ドキュメント。イメージスキャン、SBOM 生成、CI 統合の方法。
4. **OCI Distribution Specification** https://github.com/opencontainers/distribution-spec -- コンテナイメージ配布の業界標準仕様。
5. **Sigstore - cosign** https://docs.sigstore.dev/signing/signing_with_containers/ -- コンテナイメージの署名と検証。
6. **Anchore - Syft** https://github.com/anchore/syft -- SBOM 生成ツールの公式ドキュメント。
7. **dive** https://github.com/wagoodman/dive -- Docker イメージのレイヤー分析ツール。
8. **skopeo** https://github.com/containers/skopeo -- コンテナイメージのコピー・検査ツール。

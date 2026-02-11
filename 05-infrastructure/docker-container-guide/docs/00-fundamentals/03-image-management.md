# イメージ管理

> Docker イメージの取得・作成・配布からレジストリの活用まで、イメージのライフサイクル全体を管理するための実践ガイド。

---

## この章で学ぶこと

1. **Docker イメージのレイヤー構造とタグ体系**を理解し、効率的なイメージ管理ができる
2. **Docker Hub と GitHub Container Registry** の使い分けを理解し、イメージを配布できる
3. **イメージの検査・セキュリティ確認・クリーンアップ**を実践できる

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

### 比較表 1: タグ戦略

| 戦略 | 例 | 用途 | メリット | デメリット |
|---|---|---|---|---|
| セマンティック | `v1.2.3` | リリース管理 | バージョンが明確 | 手動管理が必要 |
| Git SHA | `sha-abc123f` | CI/CD | 完全なトレーサビリティ | 人間に分かりにくい |
| 日付 | `2024-01-15` | 定期ビルド | 時系列が明確 | 1日複数ビルドで衝突 |
| latest | `latest` | 開発テスト | 常に最新 | 再現性なし |
| 環境 | `production` | デプロイ管理 | 直感的 | ミュータブル |

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

# GitHub Actions からのプッシュ
# .github/workflows/publish.yml
# jobs:
#   publish:
#     permissions:
#       packages: write
#     steps:
#       - uses: docker/login-action@v3
#         with:
#           registry: ghcr.io
#           username: ${{ github.actor }}
#           password: ${{ secrets.GITHUB_TOKEN }}
#       - run: |
#           docker build -t ghcr.io/${{ github.repository }}:${{ github.sha }} .
#           docker push ghcr.io/${{ github.repository }}:${{ github.sha }}
```

### 4.3 プライベートレジストリの構築

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

# マニフェストの確認（マルチプラットフォーム対応確認）
docker manifest inspect nginx:alpine
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
+------------------------------------------------------+
```

```bash
# Docker Scout（Docker Desktop 統合）
docker scout quickview nginx:alpine
docker scout cves nginx:alpine

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

# Grype
docker run --rm anchore/grype nginx:alpine
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
cosign sign ghcr.io/myorg/my-app:v1.0.0
cosign verify ghcr.io/myorg/my-app:v1.0.0
```

---

## 8. イメージのクリーンアップ

```bash
# ローカルイメージの一覧
docker images

# 特定のイメージを削除
docker rmi nginx:alpine

# dangling イメージ（タグなし）を削除
docker image prune

# 未使用の全イメージを削除
docker image prune -a

# 特定の条件でフィルタリングして削除
docker image prune -a --filter "until=24h"  # 24時間以上前のイメージ
docker image prune -a --filter "label=env=dev"  # ラベル指定

# ディスク使用量の確認
docker system df
# TYPE            TOTAL   ACTIVE   SIZE      RECLAIMABLE
# Images          25      5        8.5GB     6.2GB (72%)
# Containers      10      3        250MB     180MB (72%)
# Local Volumes   8       4        2.1GB     900MB (42%)
# Build Cache     100     0        3.5GB     3.5GB (100%)

# ビルドキャッシュのクリーンアップ
docker builder prune

# 完全クリーンアップ
docker system prune -a --volumes
```

---

## 9. イメージの保存と転送

```bash
# イメージをファイルに保存
docker save -o my-app-v1.tar my-app:v1.0.0

# 圧縮して保存
docker save my-app:v1.0.0 | gzip > my-app-v1.tar.gz

# ファイルからイメージを読み込み
docker load -i my-app-v1.tar

# 圧縮ファイルから読み込み
gunzip -c my-app-v1.tar.gz | docker load

# コンテナの現在の状態をイメージとして保存
docker commit my-container my-app:snapshot
# 注意: commit は開発では非推奨。Dockerfile を使うべき。
```

---

## 10. アンチパターン

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

---

## 11. FAQ

### Q1: `docker pull` は毎回イメージ全体をダウンロードするのですか？

**A:** いいえ。Docker はレイヤー単位でダウンロードし、ローカルに既存のレイヤーはスキップする。そのため、ベースイメージが同じ場合、2 回目以降のプルは差分のみで非常に高速になる。`docker pull` の出力で `Already exists` と表示されるレイヤーはキャッシュが使われている。

### Q2: Docker Hub の Rate Limit はどの程度ですか？

**A:** 匿名ユーザーは 6 時間あたり 100 プル、無料アカウントは 6 時間あたり 200 プルの制限がある。CI/CD で頻繁にプルする場合は、Docker Hub の有料プランか、GitHub Container Registry 等の代替レジストリの利用を検討すべきである。ミラーレジストリを構築して Rate Limit を回避する方法もある。

### Q3: マルチプラットフォームイメージとは何ですか？

**A:** 1 つのタグで複数のアーキテクチャ（amd64, arm64 等）に対応するイメージのことである。`docker manifest` で管理され、`docker pull` 時にホストのアーキテクチャに合ったイメージが自動的に選択される。`docker buildx build --platform linux/amd64,linux/arm64` で作成できる。

### Q4: イメージサイズを確認する方法は？

**A:** `docker images` でローカルのサイズを、`docker manifest inspect` でレジストリ上のサイズを確認できる。ただし、`docker images` の SIZE はレイヤー共有を考慮しない見かけのサイズである。実際のディスク使用量は `docker system df -v` で確認する。

---

## 12. まとめ

| 項目 | ポイント |
|---|---|
| レイヤー構造 | イメージはレイヤーの積み重ね。変更されないレイヤーはキャッシュ再利用 |
| タグ | セマンティックバージョニング推奨。latest 依存は避ける |
| レジストリ | Docker Hub, GHCR, ECR 等。用途とコストで選択 |
| セキュリティ | Trivy / Docker Scout で定期的に脆弱性スキャン |
| クリーンアップ | `docker system prune` で定期的にディスクを解放 |
| 保存・転送 | `docker save/load` でオフライン環境にも対応 |
| ベストプラクティス | Dockerfile で管理、commit は非推奨 |

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

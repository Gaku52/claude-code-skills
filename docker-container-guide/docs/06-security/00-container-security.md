# コンテナセキュリティ (Container Security)

> イメージスキャン (Trivy)、最小権限原則、シークレット管理を軸に、コンテナ環境のセキュリティを多層的に強化する手法を体系的に学ぶ。

## この章で学ぶこと

1. **イメージスキャンによる脆弱性検出** -- Trivy を中心としたスキャンツールで、コンテナイメージの既知脆弱性を CI/CD パイプラインで自動検出する
2. **最小権限コンテナの構築** -- 非 root ユーザー、読み取り専用ファイルシステム、Capability の制限により攻撃面を最小化する
3. **シークレット管理と実行時セキュリティ** -- 機密情報の安全な注入と、実行時の異常検知・防御戦略を理解する

---

## 1. コンテナセキュリティの多層防御

```
+------------------------------------------------------------------+
|              コンテナセキュリティの多層防御モデル                     |
+------------------------------------------------------------------+
|                                                                  |
|  Layer 1: イメージセキュリティ                                    |
|    +-- ベースイメージの選択 (最小イメージ)                         |
|    +-- 脆弱性スキャン (Trivy, Grype)                             |
|    +-- マルチステージビルド                                       |
|    +-- イメージ署名 (cosign)                                     |
|                                                                  |
|  Layer 2: ビルドセキュリティ                                      |
|    +-- Dockerfile ベストプラクティス                               |
|    +-- シークレットのビルド時排除                                  |
|    +-- CI/CD ゲート (スキャン不合格 = デプロイ拒否)               |
|                                                                  |
|  Layer 3: ランタイムセキュリティ                                   |
|    +-- 非 root ユーザー                                          |
|    +-- 読み取り専用ファイルシステム                                |
|    +-- Capability 制限                                           |
|    +-- seccomp / AppArmor プロファイル                            |
|                                                                  |
|  Layer 4: オーケストレーションセキュリティ                          |
|    +-- Pod Security Standards                                    |
|    +-- Network Policy                                            |
|    +-- RBAC                                                      |
|    +-- Secret 管理                                               |
|                                                                  |
|  Layer 5: 監視・検知                                              |
|    +-- ログ監査                                                  |
|    +-- 異常検知 (Falco)                                          |
|    +-- イメージポリシー (OPA/Gatekeeper)                         |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 2. イメージスキャン (Trivy)

### 2.1 Trivy の基本使用

```bash
# イメージスキャン
trivy image myapp:latest

# 重大度フィルタ (CRITICAL と HIGH のみ)
trivy image --severity CRITICAL,HIGH myapp:latest

# JSON 出力 (CI 用)
trivy image --format json --output results.json myapp:latest

# Dockerfile スキャン (設定ミス検出)
trivy config Dockerfile

# ファイルシステムスキャン (ローカルプロジェクト)
trivy fs --scanners vuln,secret .

# SBOM 生成
trivy image --format spdx-json --output sbom.json myapp:latest
```

### 2.2 CI/CD への統合

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  trivy-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'           # 脆弱性が見つかったら失敗

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

  trivy-config:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Scan Dockerfile for misconfigurations
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'config'
          scan-ref: '.'
          exit-code: '1'
          severity: 'CRITICAL,HIGH'
```

### 2.3 スキャンツール比較

| ツール | 開発元 | スキャン対象 | 速度 | DB 更新頻度 | OSS |
|-------|-------|------------|------|-----------|-----|
| Trivy | Aqua Security | イメージ/FS/IaC/Secret | 高速 | 毎日 | Yes |
| Grype | Anchore | イメージ/FS | 高速 | 毎日 | Yes |
| Snyk | Snyk | イメージ/コード/IaC | 中 | リアルタイム | Freemium |
| Docker Scout | Docker | イメージ | 中 | 毎日 | Freemium |
| Clair | CoreOS/RedHat | イメージ | 遅い | 毎日 | Yes |

---

## 3. 安全な Dockerfile の構築

### 3.1 セキュアな Dockerfile

```dockerfile
# Dockerfile (セキュリティ強化版)

# ---- ビルドステージ ----
FROM node:20-alpine AS builder

WORKDIR /app

# 依存関係のみ先にコピー (キャッシュ効率)
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile

COPY . .
RUN pnpm build

# 不要ファイルの削除
RUN pnpm prune --production && \
    rm -rf .git .env* *.md tests/ src/

# ---- 本番ステージ ----
FROM node:20-alpine AS production

# セキュリティアップデート
RUN apk update && apk upgrade && \
    apk add --no-cache dumb-init && \
    rm -rf /var/cache/apk/*

# 非 root ユーザーの作成
RUN addgroup -g 1001 -S nodejs && \
    adduser -S appuser -u 1001 -G nodejs

WORKDIR /app

# ビルド成果物のみコピー
COPY --from=builder --chown=appuser:nodejs /app/dist ./dist
COPY --from=builder --chown=appuser:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=appuser:nodejs /app/package.json ./

# 非 root ユーザーに切り替え
USER appuser

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => { process.exit(r.statusCode === 200 ? 0 : 1) })"

# 読み取り専用を示唆 (実行時に --read-only で強制)
VOLUME ["/tmp"]

EXPOSE 3000

# PID 1 問題の回避 (シグナルの適切な処理)
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/index.js"]
```

### 3.2 ベースイメージの選択

```
+------------------------------------------------------------------+
|              ベースイメージのサイズとセキュリティ                     |
+------------------------------------------------------------------+
|                                                                  |
|  イメージ              | サイズ   | CVE数(参考) | 用途           |
|  ----------------------|---------|------------|---------------|
|  ubuntu:22.04          | ~77MB   | 中         | 汎用           |
|  debian:bookworm-slim  | ~74MB   | 中         | 汎用           |
|  node:20-bookworm      | ~350MB  | 多         | 開発用         |
|  node:20-alpine        | ~130MB  | 少         | 本番推奨       |
|  node:20-slim          | ~180MB  | 中         | Alpine非互換時  |
|  gcr.io/distroless/    | ~20MB   | 最少       | 本番最適       |
|  scratch               | 0MB     | なし       | Go/Rust静的バイナリ |
|                                                                  |
|  推奨: alpine (Node.js) / distroless (Go/Java) / scratch (Rust) |
|                                                                  |
+------------------------------------------------------------------+
```

### 3.3 Distroless イメージの利用

```dockerfile
# Go アプリの Distroless ビルド
FROM golang:1.22 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /server .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=builder /server /server
USER nonroot:nonroot
EXPOSE 8080
ENTRYPOINT ["/server"]
```

---

## 4. 最小権限の実行

### 4.1 Docker run のセキュリティオプション

```bash
# セキュリティ強化された docker run
docker run \
  --read-only \                        # 読み取り専用ファイルシステム
  --tmpfs /tmp:noexec,nosuid,size=64m \ # 書き込み可能な一時領域
  --cap-drop ALL \                     # 全 Capability を削除
  --cap-add NET_BIND_SERVICE \         # 必要な Capability のみ追加
  --security-opt no-new-privileges \   # 権限昇格を防止
  --security-opt seccomp=default \     # seccomp プロファイル
  --user 1001:1001 \                   # 非 root ユーザー
  --pids-limit 100 \                   # プロセス数制限
  --memory 256m \                      # メモリ制限
  --cpus 0.5 \                         # CPU 制限
  myapp:latest
```

### 4.2 Docker Compose での設定

```yaml
services:
  app:
    image: myapp:latest
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=64m
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    security_opt:
      - no-new-privileges:true
    user: "1001:1001"
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 256M
          pids: 100
```

### 4.3 Kubernetes Pod Security Standards

```yaml
# pod-security.yaml (Restricted レベル)
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1001
    runAsGroup: 1001
    fsGroup: 1001
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: myapp:latest
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
      volumeMounts:
        - name: tmp
          mountPath: /tmp
      resources:
        limits:
          cpu: "500m"
          memory: "256Mi"
        requests:
          cpu: "100m"
          memory: "128Mi"
  volumes:
    - name: tmp
      emptyDir:
        sizeLimit: 64Mi
```

---

## 5. シークレット管理

### 5.1 シークレット管理の比較

| 方式 | セキュリティ | 複雑度 | コスト | 適用場面 |
|------|-----------|-------|-------|---------|
| 環境変数 (直接) | 低 | 低 | 無料 | 開発環境のみ |
| Docker Secrets | 中 | 低 | 無料 | Docker Swarm |
| .env ファイル | 低 | 低 | 無料 | ローカル開発 |
| HashiCorp Vault | 高 | 高 | 有料/OSS | エンタープライズ |
| AWS Secrets Manager | 高 | 中 | 従量課金 | AWS 環境 |
| External Secrets | 高 | 中 | 連携先依存 | Kubernetes |
| Sealed Secrets | 中 | 低 | 無料 | GitOps |
| SOPS | 中 | 低 | 無料 | GitOps |

### 5.2 ビルド時のシークレット

```dockerfile
# NG: シークレットをビルド引数で渡す (レイヤーに残る)
ARG NPM_TOKEN
RUN echo "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > .npmrc && \
    npm ci && \
    rm .npmrc   # 削除しても前のレイヤーに残っている！

# OK: BuildKit のシークレットマウント
# syntax=docker/dockerfile:1
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc \
    npm ci
```

```bash
# ビルド時のシークレット渡し
docker build --secret id=npmrc,src=$HOME/.npmrc -t myapp .
```

### 5.3 実行時のシークレット注入

```yaml
# Docker Compose でのシークレット
services:
  app:
    image: myapp:latest
    secrets:
      - db_password
      - api_key
    environment:
      DB_PASSWORD_FILE: /run/secrets/db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    environment: API_KEY
```

```typescript
// アプリ側: ファイルベースのシークレット読み取り
import { readFileSync } from 'fs';

function getSecret(name: string): string {
  const filePath = process.env[`${name}_FILE`];
  if (filePath) {
    return readFileSync(filePath, 'utf-8').trim();
  }
  return process.env[name] || '';
}

const dbPassword = getSecret('DB_PASSWORD');
```

---

## 6. セキュリティスキャンの自動化フロー

```
+------------------------------------------------------------------+
|              セキュリティスキャン自動化フロー                        |
+------------------------------------------------------------------+
|                                                                  |
|  [開発者]                                                        |
|     | git push                                                   |
|     v                                                            |
|  [CI/CD]                                                         |
|     |                                                            |
|     +-- (1) Dockerfile Lint (hadolint)                           |
|     |     → Dockerfile のベストプラクティス違反を検出              |
|     |                                                            |
|     +-- (2) 依存関係スキャン (npm audit / Trivy fs)              |
|     |     → パッケージの既知脆弱性を検出                          |
|     |                                                            |
|     +-- (3) シークレットスキャン (Trivy / gitleaks)              |
|     |     → ハードコードされた認証情報を検出                      |
|     |                                                            |
|     +-- (4) イメージビルド                                       |
|     |                                                            |
|     +-- (5) イメージスキャン (Trivy image)                       |
|     |     → CRITICAL/HIGH → ビルド失敗                           |
|     |                                                            |
|     +-- (6) イメージ署名 (cosign)                                |
|     |                                                            |
|     +-- (7) レジストリにプッシュ                                  |
|                                                                  |
+------------------------------------------------------------------+
```

---

## アンチパターン

### アンチパターン 1: root ユーザーでのコンテナ実行

```dockerfile
# NG: root で実行 (デフォルト)
FROM node:20-alpine
WORKDIR /app
COPY . .
CMD ["node", "index.js"]
# → コンテナ内で root 権限。脆弱性を突かれると
#   ホストファイルシステムにアクセスされるリスク

# OK: 非 root ユーザーで実行
FROM node:20-alpine
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup
WORKDIR /app
COPY --chown=appuser:appgroup . .
USER appuser
CMD ["node", "index.js"]
```

**問題点**: root でコンテナを実行すると、コンテナ脱出の脆弱性 (CVE-2024-21626 等) を突かれた場合にホストの root 権限を奪取される。非 root ユーザーで実行するだけで攻撃の影響を大幅に軽減できる。

### アンチパターン 2: マルチステージビルドを使わない

```dockerfile
# NG: 単一ステージ (ビルドツール + ソースコードが本番イメージに残る)
FROM node:20
WORKDIR /app
COPY . .
RUN npm ci && npm run build
CMD ["node", "dist/index.js"]
# → gcc, make, python, .git, src/ が全て残る (攻撃面が広い)

# OK: マルチステージで本番イメージを最小化
FROM node:20 AS builder
WORKDIR /app
COPY . .
RUN npm ci && npm run build && npm prune --production

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
USER node
CMD ["node", "dist/index.js"]
```

**問題点**: ビルドツール (gcc, make) やソースコード、テストファイルが本番イメージに含まれると、脆弱性の攻撃面が不必要に広がる。マルチステージビルドで実行に必要なファイルだけを最終イメージにコピーする。

---

## FAQ

### Q1: Trivy のスキャン結果で CRITICAL が出たが、すぐに修正できない場合はどうすべきですか？

**A**: (1) `.trivyignore` ファイルに CVE ID を記載して一時的にスキップし、チケットを作成して追跡する。(2) ベースイメージを更新して修正版が含まれるか確認する。(3) 該当パッケージが実際にアプリで使用されているか確認する (到達可能性分析)。Trivy の `--ignore-unfixed` オプションで修正版がリリースされていない脆弱性を除外することも有効。重要なのは「無視する」のではなく「追跡する」こと。

### Q2: Distroless イメージにシェルがないのですが、デバッグはどうすればよいですか？

**A**: (1) `gcr.io/distroless/base-debian12:debug` タグにはシェル (busybox) が含まれている。ステージング環境では debug タグを使い、本番では通常タグを使う。(2) Kubernetes では `kubectl debug` でエフェメラルコンテナをアタッチできる。(3) `docker exec` の代わりに `docker cp` でファイルを取り出して確認する。本番でデバッグ用ツールを排除するのはセキュリティ上重要。

### Q3: read_only ファイルシステムで動作しないアプリへの対処法は？

**A**: 多くのアプリは `/tmp` や特定ディレクトリへの書き込みを必要とする。`tmpfs` で必要なパスだけを書き込み可能にする。Node.js の場合は `/tmp` と `/app/.cache`、Python の場合は `/tmp` と `__pycache__`、Nginx の場合は `/var/cache/nginx` と `/var/run` を tmpfs にマウントする。書き込み先を特定するには、`strace` やアプリのエラーログで `EROFS (Read-only file system)` を検索するとよい。

---

## まとめ

| 項目 | 要点 |
|------|------|
| イメージスキャン | Trivy を CI/CD に統合。CRITICAL/HIGH で自動ブロック |
| ベースイメージ | alpine / distroless / scratch を用途に応じて選択 |
| マルチステージ | ビルドツールとソースコードを本番イメージから排除 |
| 非 root 実行 | `USER` で非 root ユーザーを指定。UID 1001+ を使用 |
| 読み取り専用 | `read_only: true` + `tmpfs` で書き込みを最小限に |
| Capability | `cap_drop: ALL` + 必要最小限の `cap_add` |
| シークレット | BuildKit secret mount / Docker Secrets / External Secrets |
| 監視 | Falco / OPA Gatekeeper で実行時の異常を検知 |

## 次に読むべきガイド

- [サプライチェーンセキュリティ](./01-supply-chain-security.md) -- イメージ署名 (cosign) と SBOM
- [Kubernetes 応用](../05-orchestration/02-kubernetes-advanced.md) -- Pod Security Standards と Network Policy
- Docker Compose 応用 -- セキュリティ設定を含む Compose 構成

## 参考文献

1. **Trivy 公式ドキュメント** -- https://aquasecurity.github.io/trivy/ -- Trivy のインストール・設定・CI 統合の完全ガイド
2. **Docker セキュリティベストプラクティス** -- https://docs.docker.com/build/building/best-practices/ -- 公式が推奨するセキュアな Dockerfile の書き方
3. **CIS Docker Benchmark** -- https://www.cisecurity.org/benchmark/docker -- Docker セキュリティの業界標準ベンチマーク
4. **NIST SP 800-190** -- https://csrc.nist.gov/publications/detail/sp/800-190/final -- コンテナセキュリティに関する NIST ガイドライン

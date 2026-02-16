# コンテナセキュリティ (Container Security)

> イメージスキャン (Trivy)、最小権限原則、シークレット管理を軸に、コンテナ環境のセキュリティを多層的に強化する手法を体系的に学ぶ。

## この章で学ぶこと

1. **イメージスキャンによる脆弱性検出** -- Trivy を中心としたスキャンツールで、コンテナイメージの既知脆弱性を CI/CD パイプラインで自動検出する
2. **最小権限コンテナの構築** -- 非 root ユーザー、読み取り専用ファイルシステム、Capability の制限により攻撃面を最小化する
3. **シークレット管理と実行時セキュリティ** -- 機密情報の安全な注入と、実行時の異常検知・防御戦略を理解する
4. **Kubernetes Pod Security Standards** -- Pod レベルのセキュリティポリシーを適用し、クラスタ全体のセキュリティベースラインを確立する
5. **ランタイムセキュリティ監視** -- Falco を使ったリアルタイム異常検知と OPA/Gatekeeper によるポリシー適用

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
|    +-- Dockerfile Lint (hadolint)                                |
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
|    +-- ServiceAccount トークンの自動マウント無効化                  |
|                                                                  |
|  Layer 5: 監視・検知                                              |
|    +-- ログ監査                                                  |
|    +-- 異常検知 (Falco)                                          |
|    +-- イメージポリシー (OPA/Gatekeeper)                         |
|    +-- ネットワークトラフィック分析                                |
|                                                                  |
+------------------------------------------------------------------+
```

各レイヤーが独立した防御を提供し、一つのレイヤーが突破されても他のレイヤーで防御できる「多層防御 (Defense in Depth)」の考え方が基本となる。単一のセキュリティ対策に依存するのではなく、複数のレイヤーを組み合わせることで、攻撃者にとって突破すべき壁を増やす。

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

# テーブル形式でファイルに出力
trivy image --format table --output results.txt myapp:latest

# Dockerfile スキャン (設定ミス検出)
trivy config Dockerfile

# Kubernetes マニフェストのスキャン
trivy config --policy-bundle-repository ghcr.io/aquasecurity/trivy-policies k8s/

# ファイルシステムスキャン (ローカルプロジェクト)
trivy fs --scanners vuln,secret .

# ライセンスコンプライアンスチェック
trivy image --scanners license myapp:latest

# SBOM 生成
trivy image --format spdx-json --output sbom.json myapp:latest

# 特定の CVE を無視
trivy image --ignorefile .trivyignore myapp:latest

# 修正版がリリースされていない脆弱性を除外
trivy image --ignore-unfixed myapp:latest
```

### 2.2 .trivyignore ファイル

```text
# .trivyignore
# 修正版未リリースのため一時的に無視 (2025-06-01 まで追跡)
CVE-2024-12345

# アプリで使用していないパッケージの脆弱性
CVE-2024-67890  # libxml2 - このアプリでは XML パースを行わない

# テスト環境のみで使用するパッケージ
CVE-2024-11111  # dev dependency のみ
```

### 2.3 CI/CD への統合

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    # 毎日深夜に定期スキャン (新しい CVE の検出)
    - cron: '0 0 * * *'

jobs:
  hadolint:
    name: Dockerfile Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run hadolint
        uses: hadolint/hadolint-action@v3
        with:
          dockerfile: Dockerfile
          failure-threshold: warning

  trivy-scan:
    name: Image Vulnerability Scan
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
          ignore-unfixed: true

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

  trivy-config:
    name: Configuration Scan
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

  trivy-fs:
    name: Filesystem & Secret Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Scan for vulnerabilities and secrets
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          scanners: 'vuln,secret'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'
```

### 2.4 hadolint による Dockerfile リント

```bash
# hadolint のインストール
# macOS
brew install hadolint

# Docker
docker run --rm -i hadolint/hadolint < Dockerfile

# 設定ファイル (.hadolint.yaml)
```

```yaml
# .hadolint.yaml
ignored:
  - DL3008  # apt-get でバージョン固定しない (alpine では不要)
  - DL3018  # apk でバージョン固定しない

trustedRegistries:
  - docker.io
  - ghcr.io
  - gcr.io

override:
  error:
    - DL3000  # WORKDIR は絶対パスを使う
    - DL3001  # パイプに関する注意
  warning:
    - DL3042  # pip install に --no-cache-dir を使う
  info:
    - DL3059  # 複数の連続する RUN 命令を統合する
```

### 2.5 スキャンツール比較

| ツール | 開発元 | スキャン対象 | 速度 | DB 更新頻度 | OSS |
|-------|-------|------------|------|-----------|-----|
| Trivy | Aqua Security | イメージ/FS/IaC/Secret | 高速 | 毎日 | Yes |
| Grype | Anchore | イメージ/FS | 高速 | 毎日 | Yes |
| Snyk | Snyk | イメージ/コード/IaC | 中 | リアルタイム | Freemium |
| Docker Scout | Docker | イメージ | 中 | 毎日 | Freemium |
| Clair | CoreOS/RedHat | イメージ | 遅い | 毎日 | Yes |

### 2.6 Grype の使用例

```bash
# Grype のインストール
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin

# イメージスキャン
grype myapp:latest

# SBOM からスキャン
grype sbom:sbom.json

# JSON 出力
grype myapp:latest -o json > grype-results.json

# 重大度でフィルタ
grype myapp:latest --fail-on critical
```

---

## 3. 安全な Dockerfile の構築

### 3.1 セキュアな Dockerfile (Node.js)

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

### 3.2 セキュアな Dockerfile (Python)

```dockerfile
# Dockerfile (Python セキュリティ強化版)

# ---- ビルドステージ ----
FROM python:3.12-slim AS builder

WORKDIR /app

# 仮想環境を作成 (システム Python との分離)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# ---- 本番ステージ ----
FROM python:3.12-slim AS production

# セキュリティアップデート
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends tini && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 非 root ユーザー
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -s /bin/false -M appuser

WORKDIR /app

# 仮想環境をコピー
COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appgroup /app .

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

ENTRYPOINT ["tini", "--"]
CMD ["gunicorn", "app.main:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
```

### 3.3 セキュアな Dockerfile (Go)

```dockerfile
# Dockerfile (Go セキュリティ強化版)

# ---- ビルドステージ ----
FROM golang:1.22-alpine AS builder

RUN apk add --no-cache ca-certificates git

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download && go mod verify

COPY . .

# 静的バイナリを生成 (CGO 無効)
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s -X main.version=1.0.0" \
    -o /server ./cmd/server

# ---- 本番ステージ (scratch or distroless) ----
FROM gcr.io/distroless/static-debian12:nonroot

# CA 証明書 (HTTPS 通信に必要)
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# バイナリのみコピー
COPY --from=builder /server /server

USER nonroot:nonroot

EXPOSE 8080

ENTRYPOINT ["/server"]
```

### 3.4 ベースイメージの選択

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
|  python:3.12-slim      | ~120MB  | 少         | Python 本番    |
|  python:3.12-alpine    | ~50MB   | 最少       | Alpine 互換時  |
|  golang:1.22-alpine    | ~250MB  | 少         | Go ビルド用    |
|  gcr.io/distroless/    | ~20MB   | 最少       | 本番最適       |
|  chainguard/           | ~10MB   | 最少       | 本番最適       |
|  scratch               | 0MB     | なし       | Go/Rust静的バイナリ |
|                                                                  |
|  推奨: alpine (Node.js) / distroless (Go/Java) / scratch (Rust) |
|                                                                  |
+------------------------------------------------------------------+
```

ベースイメージの選択は、セキュリティとイメージサイズの両方に影響する。イメージに含まれるパッケージが多いほど CVE (既知脆弱性) の数も増える。最小イメージを使うことで攻撃面を縮小し、スキャン結果のノイズも減らせる。

### 3.5 Distroless イメージの利用

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

```dockerfile
# Java アプリの Distroless ビルド
FROM eclipse-temurin:21-jdk AS builder
WORKDIR /app
COPY . .
RUN ./gradlew bootJar

FROM gcr.io/distroless/java21-debian12:nonroot
COPY --from=builder /app/build/libs/app.jar /app.jar
USER nonroot:nonroot
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app.jar"]
```

### 3.6 Chainguard Images (次世代の最小イメージ)

```dockerfile
# Chainguard Images を使った Node.js アプリ
FROM cgr.dev/chainguard/node:latest-dev AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --production
COPY . .
RUN npm run build

FROM cgr.dev/chainguard/node:latest
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
EXPOSE 3000
CMD ["dist/index.js"]
```

Chainguard Images は Wolfi Linux をベースとし、CVE ゼロを目標とする最小イメージ群。`apk` パッケージマネージャを使い、Alpine と互換性がある。Distroless の後継として注目されている。

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
  --network myapp-net \                # カスタムネットワーク (default bridge を使わない)
  --dns 8.8.8.8 \                      # DNS サーバーの明示指定
  --health-cmd "curl -f http://localhost:3000/health || exit 1" \
  --health-interval 30s \
  --health-timeout 5s \
  --health-retries 3 \
  myapp:latest
```

### 4.2 Docker Compose での設定

```yaml
# docker-compose.yml
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
    healthcheck:
      test: ["CMD", "node", "-e", "require('http').get('http://localhost:3000/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    networks:
      - app-net
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

networks:
  app-net:
    driver: bridge
    internal: false  # true にすると外部通信を遮断
```

### 4.3 Linux Capability の詳細

```
+------------------------------------------------------------------+
|              主要な Linux Capability                               |
+------------------------------------------------------------------+
|                                                                  |
|  Capability            | 説明                    | 必要な場面      |
|  ----------------------|------------------------|----------------|
|  NET_BIND_SERVICE      | 1024未満のポートにバインド | Nginx (80/443)  |
|  NET_RAW               | RAW ソケット作成         | ping コマンド    |
|  CHOWN                 | ファイル所有者の変更      | 初期化スクリプト  |
|  DAC_OVERRIDE          | ファイルパーミッション無視 | 特権操作         |
|  SETUID/SETGID         | UID/GID の変更           | su / sudo       |
|  SYS_ADMIN             | 広範な管理権限            | マウント操作      |
|  SYS_PTRACE            | プロセストレース          | デバッグ          |
|  SYS_TIME              | システム時刻変更          | NTP クライアント  |
|  AUDIT_WRITE           | 監査ログ書き込み          | sshd             |
|  KILL                  | シグナル送信             | プロセス管理      |
|                                                                  |
|  デフォルト: Docker は 14 個の Capability を付与                   |
|  推奨: cap_drop ALL + 必要最小限の cap_add                        |
|                                                                  |
+------------------------------------------------------------------+
```

### 4.4 seccomp プロファイル

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "archMap": [
    {
      "architecture": "SCMP_ARCH_X86_64",
      "subArchitectures": ["SCMP_ARCH_X86"]
    }
  ],
  "syscalls": [
    {
      "names": [
        "accept", "accept4", "access", "bind", "brk",
        "chdir", "chmod", "chown", "close", "connect",
        "dup", "dup2", "dup3", "epoll_create", "epoll_create1",
        "epoll_ctl", "epoll_wait", "epoll_pwait",
        "execve", "exit", "exit_group",
        "fchmod", "fchown", "fcntl", "fdatasync",
        "fstat", "fstatfs", "fsync", "ftruncate",
        "getcwd", "getdents", "getdents64", "getegid",
        "geteuid", "getgid", "getpgrp", "getpid", "getppid",
        "getuid", "ioctl", "kill",
        "listen", "lseek", "lstat",
        "madvise", "mkdir", "mmap", "mprotect", "mremap",
        "munmap", "nanosleep", "newfstatat",
        "open", "openat", "pipe", "pipe2", "poll", "ppoll",
        "prctl", "pread64", "prlimit64", "pwrite64",
        "read", "readlink", "readlinkat", "recvfrom", "recvmsg",
        "rename", "rmdir", "rt_sigaction", "rt_sigprocmask",
        "rt_sigreturn", "select", "sendmsg", "sendto",
        "set_robust_list", "set_tid_address",
        "setgid", "setgroups", "setuid",
        "sigaltstack", "socket", "stat", "statfs",
        "symlink", "tgkill", "umask", "uname",
        "unlink", "wait4", "write", "writev"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

```bash
# カスタム seccomp プロファイルの適用
docker run --security-opt seccomp=seccomp-profile.json myapp:latest
```

### 4.5 Kubernetes Pod Security Standards

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
  automountServiceAccountToken: false  # SA トークンの自動マウントを無効化
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

### 4.6 Pod Security Standards (PSS) の適用

```yaml
# Namespace レベルで Pod Security Standards を適用
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    # Restricted レベルを強制
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest
    # Baseline レベルで警告
    pod-security.kubernetes.io/warn: baseline
    pod-security.kubernetes.io/warn-version: latest
    # Privileged レベルで監査ログ
    pod-security.kubernetes.io/audit: privileged
    pod-security.kubernetes.io/audit-version: latest
```

```
+------------------------------------------------------------------+
|              Pod Security Standards レベル                         |
+------------------------------------------------------------------+
|                                                                  |
|  Privileged (特権):                                              |
|    -> 制限なし。全ての設定が許可される                              |
|    -> 用途: kube-system, 監視エージェント                          |
|                                                                  |
|  Baseline (基準):                                                 |
|    -> 既知の特権昇格を防止する最小限の制限                          |
|    -> 禁止: hostNetwork, hostPID, hostIPC, 特権コンテナ            |
|    -> 禁止: hostPath ボリューム (一部)                              |
|    -> 用途: ステージング環境、開発環境                              |
|                                                                  |
|  Restricted (制限):                                               |
|    -> 現在のベストプラクティスに沿った厳格な制限                    |
|    -> 追加要件: runAsNonRoot, readOnlyRootFilesystem              |
|    -> 追加要件: allowPrivilegeEscalation: false                    |
|    -> 追加要件: capabilities drop ALL, seccomp RuntimeDefault      |
|    -> 用途: 本番環境                                               |
|                                                                  |
+------------------------------------------------------------------+
```

### 4.7 RBAC (ロールベースアクセス制御)

```yaml
# ServiceAccount
apiVersion: v1
kind: ServiceAccount
metadata:
  name: myapp-sa
  namespace: production
automountServiceAccountToken: false

---
# Role (namespace スコープ)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: myapp-role
  namespace: production
rules:
  # ConfigMap の読み取りのみ許可
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "watch"]
  # Secret の読み取りのみ許可 (特定名のみ)
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["myapp-secret"]
    verbs: ["get"]

---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: myapp-rolebinding
  namespace: production
subjects:
  - kind: ServiceAccount
    name: myapp-sa
    namespace: production
roleRef:
  kind: Role
  name: myapp-role
  apiGroup: rbac.authorization.k8s.io
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
| GCP Secret Manager | 高 | 中 | 従量課金 | GCP 環境 |
| Azure Key Vault | 高 | 中 | 従量課金 | Azure 環境 |
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

# OK: SSH 鍵のマウント (プライベート Git リポジトリのクローン用)
RUN --mount=type=ssh \
    git clone git@github.com:myorg/private-repo.git
```

```bash
# ビルド時のシークレット渡し
docker build --secret id=npmrc,src=$HOME/.npmrc -t myapp .

# SSH 鍵の転送
docker build --ssh default -t myapp .

# BuildKit のシークレットマウントを使った pip install
docker build \
  --secret id=pip_conf,src=$HOME/.pip/pip.conf \
  -t myapp .
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
import { readFileSync, existsSync } from 'fs';

function getSecret(name: string): string {
  // Docker Secrets (ファイルベース) を優先
  const filePath = process.env[`${name}_FILE`];
  if (filePath && existsSync(filePath)) {
    return readFileSync(filePath, 'utf-8').trim();
  }

  // Kubernetes Secret (環境変数) にフォールバック
  const envValue = process.env[name];
  if (envValue) {
    return envValue;
  }

  throw new Error(`Secret '${name}' not found`);
}

const dbPassword = getSecret('DB_PASSWORD');
const apiKey = getSecret('API_KEY');
```

### 5.4 HashiCorp Vault との統合

```yaml
# Vault Agent Injector を使った Kubernetes 統合
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  template:
    metadata:
      annotations:
        # Vault Agent Injector のアノテーション
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "myapp"
        vault.hashicorp.com/agent-inject-secret-db-creds: "secret/data/myapp/db"
        vault.hashicorp.com/agent-inject-template-db-creds: |
          {{- with secret "secret/data/myapp/db" -}}
          export DATABASE_URL="postgresql://{{ .Data.data.username }}:{{ .Data.data.password }}@db:5432/myapp"
          {{- end -}}
    spec:
      serviceAccountName: myapp-sa
      containers:
        - name: app
          image: myapp:latest
          command: ["/bin/sh", "-c"]
          args:
            - source /vault/secrets/db-creds && exec node dist/index.js
```

---

## 6. ランタイムセキュリティ

### 6.1 Falco による異常検知

```yaml
# Falco のインストール (Helm)
# helm install falco falcosecurity/falco -n falco --create-namespace

# falco-rules.yaml (カスタムルール)
- rule: Container Shell Spawned
  desc: コンテナ内でシェルが起動された
  condition: >
    spawned_process and
    container and
    proc.name in (bash, sh, zsh, ash) and
    not container.image.repository in (allowed_shell_images)
  output: >
    Shell spawned in container
    (user=%user.name container=%container.name
     image=%container.image.repository cmd=%proc.cmdline)
  priority: WARNING
  tags: [container, shell]

- rule: Sensitive File Access
  desc: 機密ファイルへのアクセスを検知
  condition: >
    open_read and
    container and
    fd.name in (/etc/shadow, /etc/passwd, /etc/sudoers)
  output: >
    Sensitive file accessed in container
    (user=%user.name file=%fd.name container=%container.name)
  priority: CRITICAL
  tags: [container, filesystem]

- rule: Outbound Connection to Suspicious Port
  desc: 不審なポートへの外部接続
  condition: >
    outbound and
    container and
    not fd.sport in (80, 443, 53, 5432, 6379, 9092) and
    not container.image.repository in (allowed_outbound_images)
  output: >
    Unexpected outbound connection
    (user=%user.name container=%container.name
     connection=%fd.name port=%fd.sport)
  priority: WARNING
  tags: [container, network]

- rule: Package Manager Execution
  desc: コンテナ内でパッケージマネージャが実行された
  condition: >
    spawned_process and
    container and
    proc.name in (apt, apt-get, yum, dnf, apk, pip, npm) and
    not container.image.repository in (allowed_package_install_images)
  output: >
    Package manager executed in container
    (user=%user.name cmd=%proc.cmdline container=%container.name)
  priority: ERROR
  tags: [container, software_mgmt]
```

### 6.2 Falco と Slack 連携 (アラート通知)

```yaml
# falco-values.yaml (Helm)
falcosidekick:
  enabled: true
  config:
    slack:
      webhookurl: "https://hooks.slack.com/services/T00000/B00000/XXXXXX"
      channel: "#security-alerts"
      minimumpriority: "warning"
      messageformat: |
        *Priority:* {{ .Priority }}
        *Rule:* {{ .Rule }}
        *Output:* {{ .Output }}
        *Time:* {{ .Time }}
```

### 6.3 OPA/Gatekeeper によるポリシー適用

```yaml
# Gatekeeper のインストール
# helm install gatekeeper gatekeeper/gatekeeper -n gatekeeper-system --create-namespace

# ConstraintTemplate: コンテナは root で実行してはならない
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8srequirenonrootuser
spec:
  crd:
    spec:
      names:
        kind: K8sRequireNonRootUser
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package requirenonrootuser

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not container.securityContext.runAsNonRoot
          msg := sprintf("Container '%v' must set securityContext.runAsNonRoot to true", [container.name])
        }

        violation[{"msg": msg}] {
          input.review.object.spec.securityContext.runAsUser == 0
          msg := "Pod must not run as root (UID 0)"
        }

---
# Constraint: 適用
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequireNonRootUser
metadata:
  name: require-non-root
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    namespaces: ["production", "staging"]
  parameters: {}
```

```yaml
# ConstraintTemplate: リソースリミットの必須化
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8srequireresourcelimits
spec:
  crd:
    spec:
      names:
        kind: K8sRequireResourceLimits
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package requireresourcelimits

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not container.resources.limits.cpu
          msg := sprintf("Container '%v' must set resources.limits.cpu", [container.name])
        }

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not container.resources.limits.memory
          msg := sprintf("Container '%v' must set resources.limits.memory", [container.name])
        }

---
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequireResourceLimits
metadata:
  name: require-resource-limits
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    namespaces: ["production"]
```

### 6.4 Kyverno によるポリシー適用

```yaml
# Kyverno ポリシー: 非 root 実行の強制
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-run-as-non-root
spec:
  validationFailureAction: Enforce
  background: true
  rules:
    - name: check-containers
      match:
        any:
          - resources:
              kinds:
                - Pod
              namespaces:
                - production
                - staging
      validate:
        message: "Containers must run as non-root"
        pattern:
          spec:
            containers:
              - securityContext:
                  runAsNonRoot: true
                  allowPrivilegeEscalation: false

---
# Kyverno ポリシー: latest タグの禁止
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: disallow-latest-tag
spec:
  validationFailureAction: Enforce
  rules:
    - name: validate-image-tag
      match:
        any:
          - resources:
              kinds:
                - Pod
      validate:
        message: "Image tag 'latest' is not allowed. Use a specific version tag."
        pattern:
          spec:
            containers:
              - image: "!*:latest"
```

---

## 7. セキュリティスキャンの自動化フロー

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
|     |     -> Dockerfile のベストプラクティス違反を検出              |
|     |                                                            |
|     +-- (2) 依存関係スキャン (npm audit / Trivy fs)              |
|     |     -> パッケージの既知脆弱性を検出                          |
|     |                                                            |
|     +-- (3) シークレットスキャン (Trivy / gitleaks)              |
|     |     -> ハードコードされた認証情報を検出                      |
|     |                                                            |
|     +-- (4) IaC スキャン (Trivy config / tfsec)                  |
|     |     -> Kubernetes YAML / Terraform の設定ミスを検出          |
|     |                                                            |
|     +-- (5) イメージビルド                                       |
|     |                                                            |
|     +-- (6) イメージスキャン (Trivy image)                       |
|     |     -> CRITICAL/HIGH -> ビルド失敗                           |
|     |                                                            |
|     +-- (7) SBOM 生成                                            |
|     |                                                            |
|     +-- (8) イメージ署名 (cosign)                                |
|     |                                                            |
|     +-- (9) レジストリにプッシュ                                  |
|     |                                                            |
|     +-- (10) デプロイ時: Admission Controller で署名検証          |
|                                                                  |
+------------------------------------------------------------------+
```

### 7.1 gitleaks によるシークレット検出

```bash
# gitleaks のインストール
brew install gitleaks

# リポジトリスキャン
gitleaks detect --source . --verbose

# Git 履歴全体のスキャン
gitleaks detect --source . --log-opts="--all"

# pre-commit フックとして設定
gitleaks protect --staged
```

```yaml
# .gitleaks.toml (設定ファイル)
[allowlist]
  description = "Allowlisted files and patterns"
  paths = [
    '''\.gitleaks\.toml''',
    '''tests/fixtures/''',
    '''\.trivyignore''',
  ]

[[rules]]
  id = "custom-api-key"
  description = "Custom API Key Pattern"
  regex = '''(?i)api[_-]?key\s*[=:]\s*['"]([\w\-]{32,})['"']'''
  tags = ["key", "api"]
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
# -> コンテナ内で root 権限。脆弱性を突かれると
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
# -> gcc, make, python, .git, src/ が全て残る (攻撃面が広い)

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

### アンチパターン 3: Capability を削除しない

```yaml
# NG: デフォルトの Capability のまま実行
spec:
  containers:
    - name: app
      image: myapp:latest
      # securityContext が未設定 -> 14個の Capability が付与される

# OK: 全削除して必要最小限のみ追加
spec:
  containers:
    - name: app
      image: myapp:latest
      securityContext:
        capabilities:
          drop: ["ALL"]
          add: ["NET_BIND_SERVICE"]  # 80番ポート使用時のみ
```

**問題点**: Docker はデフォルトで 14 個の Linux Capability をコンテナに付与する。NET_RAW (パケット偽装)、SYS_CHROOT (chroot 脱出) など、多くのアプリでは不要な権限が含まれる。`drop: ALL` で全て削除し、必要なものだけを明示的に追加する。

### アンチパターン 4: automountServiceAccountToken を無効化しない

```yaml
# NG: SA トークンが自動マウントされる (デフォルト)
spec:
  containers:
    - name: app
      image: myapp:latest
      # /var/run/secrets/kubernetes.io/serviceaccount/token にトークンが存在
      # -> コンテナが侵害された場合、Kubernetes API にアクセスされる

# OK: 不要な場合は自動マウントを無効化
spec:
  automountServiceAccountToken: false
  containers:
    - name: app
      image: myapp:latest
```

**問題点**: ServiceAccount トークンが自動的にマウントされると、コンテナ侵害時に攻撃者が Kubernetes API にアクセスできてしまう。Kubernetes API と通信する必要がないアプリケーションでは、必ず `automountServiceAccountToken: false` を設定する。

---

## FAQ

### Q1: Trivy のスキャン結果で CRITICAL が出たが、すぐに修正できない場合はどうすべきですか？

**A**: (1) `.trivyignore` ファイルに CVE ID を記載して一時的にスキップし、チケットを作成して追跡する。(2) ベースイメージを更新して修正版が含まれるか確認する。(3) 該当パッケージが実際にアプリで使用されているか確認する (到達可能性分析)。Trivy の `--ignore-unfixed` オプションで修正版がリリースされていない脆弱性を除外することも有効。重要なのは「無視する」のではなく「追跡する」こと。

### Q2: Distroless イメージにシェルがないのですが、デバッグはどうすればよいですか？

**A**: (1) `gcr.io/distroless/base-debian12:debug` タグにはシェル (busybox) が含まれている。ステージング環境では debug タグを使い、本番では通常タグを使う。(2) Kubernetes では `kubectl debug` でエフェメラルコンテナをアタッチできる。(3) `docker exec` の代わりに `docker cp` でファイルを取り出して確認する。本番でデバッグ用ツールを排除するのはセキュリティ上重要。

### Q3: read_only ファイルシステムで動作しないアプリへの対処法は？

**A**: 多くのアプリは `/tmp` や特定ディレクトリへの書き込みを必要とする。`tmpfs` で必要なパスだけを書き込み可能にする。Node.js の場合は `/tmp` と `/app/.cache`、Python の場合は `/tmp` と `__pycache__`、Nginx の場合は `/var/cache/nginx` と `/var/run` を tmpfs にマウントする。書き込み先を特定するには、`strace` やアプリのエラーログで `EROFS (Read-only file system)` を検索するとよい。

### Q4: Pod Security Standards の Restricted レベルを適用したら既存の Pod が起動しなくなりました。段階的に移行するには？

**A**: 段階的なアプローチを推奨する。(1) まず `audit` モードで Restricted を適用し、違反の Pod を特定する (`kubectl get events --field-selector reason=FailedCreate`)。(2) 各 Pod のセキュリティコンテキストを修正する。(3) `warn` モードに変更してテストする。(4) 全 Pod が準拠したら `enforce` モードに切り替える。一度に全 Namespace に適用するのではなく、1 つの Namespace ずつ進める。

### Q5: Falco のアラートが多すぎて対応しきれません。チューニング方法は？

**A**: (1) 正当な操作による誤検知を特定し、`exceptions` でホワイトリストに追加する。(2) 優先度 (priority) を調整し、本当に重要なアラートのみ通知する。(3) CronJob やバッチ処理による定期的なアラートは、対象コンテナやイメージを `condition` から除外する。(4) falcosidekick で `minimumpriority` を設定し、WARNING 以上のみ通知する。最初は少数のルールから始めて、環境に合わせて徐々にルールを追加するのが効果的。

### Q6: コンテナイメージのセキュリティスキャンを定期的に再実行する必要がありますか？

**A**: はい、必須。新しい CVE は毎日発見される。ビルド時にスキャンをパスしたイメージでも、後から脆弱性が発見される可能性がある。推奨は (1) ビルド時のスキャン (CI/CD ゲート)、(2) デプロイ済みイメージの定期スキャン (日次)、(3) 新しい CRITICAL CVE が公開された際の緊急スキャン。SBOM を保存しておけば、イメージを再プルせずに脆弱性の影響を確認できる。

---

## まとめ

| 項目 | 要点 |
|------|------|
| イメージスキャン | Trivy を CI/CD に統合。CRITICAL/HIGH で自動ブロック |
| Dockerfile Lint | hadolint でベストプラクティス違反を早期検出 |
| ベースイメージ | alpine / distroless / scratch / chainguard を用途に応じて選択 |
| マルチステージ | ビルドツールとソースコードを本番イメージから排除 |
| 非 root 実行 | `USER` で非 root ユーザーを指定。UID 1001+ を使用 |
| 読み取り専用 | `read_only: true` + `tmpfs` で書き込みを最小限に |
| Capability | `cap_drop: ALL` + 必要最小限の `cap_add` |
| Pod Security | Restricted レベルを本番 Namespace に適用 |
| RBAC | 最小権限の ServiceAccount + Role を設定 |
| シークレット | BuildKit secret mount / Docker Secrets / External Secrets |
| ランタイム監視 | Falco で実行時の異常を検知。アラート連携 |
| ポリシー適用 | OPA Gatekeeper / Kyverno で Admission Control |
| シークレット検出 | gitleaks で Git 履歴内のシークレットを検出 |

## 次に読むべきガイド

- [サプライチェーンセキュリティ](./01-supply-chain-security.md) -- イメージ署名 (cosign) と SBOM
- [Kubernetes 応用](../05-orchestration/02-kubernetes-advanced.md) -- Pod Security Standards と Network Policy
- Docker Compose 応用 -- セキュリティ設定を含む Compose 構成

## 参考文献

1. **Trivy 公式ドキュメント** -- https://aquasecurity.github.io/trivy/ -- Trivy のインストール・設定・CI 統合の完全ガイド
2. **Docker セキュリティベストプラクティス** -- https://docs.docker.com/build/building/best-practices/ -- 公式が推奨するセキュアな Dockerfile の書き方
3. **CIS Docker Benchmark** -- https://www.cisecurity.org/benchmark/docker -- Docker セキュリティの業界標準ベンチマーク
4. **NIST SP 800-190** -- https://csrc.nist.gov/publications/detail/sp/800-190/final -- コンテナセキュリティに関する NIST ガイドライン
5. **Falco 公式ドキュメント** -- https://falco.org/docs/ -- コンテナランタイムセキュリティ監視ツール
6. **Pod Security Standards** -- https://kubernetes.io/docs/concepts/security/pod-security-standards/ -- Kubernetes 公式の Pod セキュリティ基準
7. **OPA Gatekeeper** -- https://open-policy-agent.github.io/gatekeeper/ -- Kubernetes ポリシーエンジン
8. **Kyverno** -- https://kyverno.io/docs/ -- Kubernetes ネイティブポリシー管理

# コンテナセキュリティ

> コンテナイメージのスキャン、最小権限でのランタイム保護、安全な Dockerfile の書き方、Kubernetes のセキュリティポリシーまで、コンテナ化されたアプリケーションを守るための包括的ガイド

## この章で学ぶこと

1. **コンテナの脅威モデル** — 攻撃面の分類、コンテナエスケープのメカニズム、サプライチェーン攻撃
2. **イメージセキュリティ** — ベースイメージの選択基準、マルチステージビルド、脆弱性スキャン、SBOM 生成
3. **ランタイム保護** — Linux namespaces/cgroups、seccomp、AppArmor、非 root 実行、リードオンリーファイルシステム
4. **オーケストレーションセキュリティ** — Kubernetes SecurityContext、Pod Security Standards、NetworkPolicy、RBAC
5. **イメージ署名と検証** — cosign、Sigstore、Kyverno/OPA Gatekeeper によるアドミッションコントロール
6. **ランタイムセキュリティ監視** — Falco、Tetragon による異常検知とインシデント対応

### 前提知識

- Linux の基本操作 (ファイルシステム、プロセス、ネットワーク)
- Docker の基本操作 (build, run, pull, push)
- Kubernetes の基本概念 (Pod, Deployment, Service, Namespace)
- YAML/JSON の読み書き

---

## 1. コンテナの脅威モデル

### コンテナのアーキテクチャとセキュリティ境界

```
+----------------------------------------------------------------+
|                    ホスト OS (Linux Kernel)                       |
|  +----------------------------------------------------------+  |
|  |                  コンテナランタイム                          |  |
|  |  (containerd / CRI-O)                                     |  |
|  |                                                           |  |
|  |  +----------------+  +----------------+  +-------------+  |  |
|  |  | Container A    |  | Container B    |  | Container C |  |  |
|  |  |                |  |                |  |             |  |  |
|  |  | [App Process]  |  | [App Process]  |  | [App]       |  |  |
|  |  |                |  |                |  |             |  |  |
|  |  | Namespaces:    |  | Namespaces:    |  | Namespaces: |  |  |
|  |  |  - PID         |  |  - PID         |  |  - PID      |  |  |
|  |  |  - NET         |  |  - NET         |  |  - NET      |  |  |
|  |  |  - MNT         |  |  - MNT         |  |  - MNT      |  |  |
|  |  |  - UTS         |  |  - UTS         |  |  - UTS      |  |  |
|  |  |  - IPC         |  |  - IPC         |  |  - IPC      |  |  |
|  |  |  - USER        |  |  - USER        |  |  - USER     |  |  |
|  |  |                |  |                |  |             |  |  |
|  |  | cgroups:       |  | cgroups:       |  | cgroups:    |  |  |
|  |  |  CPU/Mem/IO    |  |  CPU/Mem/IO    |  |  CPU/Mem/IO |  |  |
|  |  +----------------+  +----------------+  +-------------+  |  |
|  +----------------------------------------------------------+  |
|                                                                 |
|  共有: カーネル、/proc、/sys、一部デバイス                         |
+----------------------------------------------------------------+
```

**重要な理解**: コンテナは仮想マシンとは異なり、カーネルを共有する。この共有カーネルが攻撃面となり、カーネル脆弱性を通じたコンテナエスケープが可能になる。

### VM とコンテナの分離レベル比較

```
分離レベルの比較:

仮想マシン (VM):
+------------------+  +------------------+
|  Guest OS        |  |  Guest OS        |
|  (独自カーネル)   |  |  (独自カーネル)   |
+------------------+  +------------------+
+--------------------------------------+
|        ハイパーバイザ (Type 1/2)       |
+--------------------------------------+
|            ホスト OS / HW             |
+--------------------------------------+
→ ハードウェアレベルの分離 (強い)

コンテナ:
+----------+  +----------+  +----------+
| App A    |  | App B    |  | App C    |
| (ns/cg)  |  | (ns/cg)  |  | (ns/cg)  |
+----------+  +----------+  +----------+
+--------------------------------------+
|        共有カーネル (Linux)            |
+--------------------------------------+
|            ホスト OS / HW             |
+--------------------------------------+
→ プロセスレベルの分離 (弱い)

gVisor / Kata Containers:
+----------+  +----------+
| App A    |  | App B    |
+----------+  +----------+
+----------+  +----------+
| gVisor   |  | microVM  |
| (Sentry) |  | (QEMU)  |
+----------+  +----------+
+--------------------------------------+
|            ホスト OS / HW             |
+--------------------------------------+
→ 中間レベルの分離 (強化)
```

### 攻撃面の分類

```
+----------------------------------------------------------+
|                コンテナの攻撃面                              |
|----------------------------------------------------------|
|                                                          |
|  [イメージ層]                                              |
|  +-- ベースイメージの脆弱性 (CVE)                          |
|  +-- アプリ依存ライブラリの脆弱性                           |
|  +-- シークレットの埋め込み (ENV, COPY)                    |
|  +-- 不要パッケージの含有 (攻撃ツール化)                    |
|  +-- SETUID/SETGID バイナリの存在                         |
|                                                          |
|  [ビルド層]                                                |
|  +-- 信頼できないレジストリからの pull                      |
|  +-- タグの可変性 (latest の上書き)                        |
|  +-- CI/CD パイプラインの侵害                              |
|  +-- ビルドキャッシュ汚染                                  |
|  +-- マルチアーキテクチャイメージの差し替え                   |
|                                                          |
|  [ランタイム層]                                            |
|  +-- root 実行による権限昇格                               |
|  +-- コンテナエスケープ (kernel exploit)                   |
|  +-- 過剰な Linux capabilities                            |
|  +-- ホストパスのマウント (/var/run/docker.sock)           |
|  +-- 特権コンテナ (--privileged)                          |
|  +-- PID 1 シグナルハンドリング問題                        |
|                                                          |
|  [ネットワーク層]                                          |
|  +-- コンテナ間の無制限通信                                |
|  +-- メタデータ API への不正アクセス (169.254.169.254)      |
|  +-- DNS スプーフィング (Pod 間)                           |
|  +-- サービスメッシュのバイパス                              |
|                                                          |
|  [オーケストレーション層]                                    |
|  +-- etcd への不正アクセス                                 |
|  +-- RBAC の過剰な権限                                    |
|  +-- ServiceAccount トークンの悪用                        |
|  +-- kubelet API への直接アクセス                          |
+----------------------------------------------------------+
```

### コンテナエスケープの主要手法

```
コンテナエスケープの攻撃パス:

1. カーネル脆弱性の悪用:
   Container (root) → kernel exploit → Host root
   例: CVE-2022-0185 (Filesystem Context)
       CVE-2024-1086 (nf_tables)

2. Docker Socket マウント:
   Container → /var/run/docker.sock → docker run --privileged → Host
   (Docker-in-Docker の危険性)

3. 特権コンテナからのエスケープ:
   Container (--privileged) → mount host / → chroot → Host root

4. capabilities の悪用:
   Container (CAP_SYS_ADMIN) → mount cgroupfs → release_agent → Host

5. /proc/sys の悪用:
   Container → /proc/self/exe → ホストバイナリの上書き
   (CVE-2019-5736: runc 脆弱性)
```

### コンテナエスケープの検証コード (Go)

```go
// escape_detector.go - コンテナエスケープのリスク要因を検出するツール
package main

import (
    "fmt"
    "os"
    "strings"
    "syscall"
)

type RiskLevel int

const (
    Low RiskLevel = iota
    Medium
    High
    Critical
)

type Finding struct {
    Check    string
    Risk     RiskLevel
    Detail   string
    Remediation string
}

func main() {
    findings := []Finding{}

    // 1. root 実行チェック
    if os.Getuid() == 0 {
        findings = append(findings, Finding{
            Check:  "Running as root",
            Risk:   High,
            Detail: fmt.Sprintf("UID=%d, GID=%d", os.Getuid(), os.Getgid()),
            Remediation: "USER ディレクティブで非 root ユーザを指定する",
        })
    }

    // 2. 特権モードチェック
    if isPrivileged() {
        findings = append(findings, Finding{
            Check:  "Privileged container",
            Risk:   Critical,
            Detail: "コンテナが特権モードで実行されている",
            Remediation: "--privileged フラグを削除する",
        })
    }

    // 3. Docker socket マウントチェック
    if _, err := os.Stat("/var/run/docker.sock"); err == nil {
        findings = append(findings, Finding{
            Check:  "Docker socket mounted",
            Risk:   Critical,
            Detail: "/var/run/docker.sock がマウントされている",
            Remediation: "Docker socket のマウントを削除する",
        })
    }

    // 4. 書き込み可能なルートファイルシステム
    if isWritableRootFS() {
        findings = append(findings, Finding{
            Check:  "Writable root filesystem",
            Risk:   Medium,
            Detail: "ルートファイルシステムが書き込み可能",
            Remediation: "readOnlyRootFilesystem: true を設定する",
        })
    }

    // 5. capabilities チェック
    caps := getEffectiveCaps()
    dangerousCaps := []string{
        "CAP_SYS_ADMIN", "CAP_SYS_PTRACE", "CAP_SYS_MODULE",
        "CAP_DAC_OVERRIDE", "CAP_NET_RAW",
    }
    for _, cap := range dangerousCaps {
        if strings.Contains(caps, cap) {
            findings = append(findings, Finding{
                Check:  fmt.Sprintf("Dangerous capability: %s", cap),
                Risk:   High,
                Detail: fmt.Sprintf("capability %s が有効", cap),
                Remediation: "capabilities を drop ALL + 必要なもののみ add する",
            })
        }
    }

    // 6. ホストネットワークチェック
    if isHostNetwork() {
        findings = append(findings, Finding{
            Check:  "Host network mode",
            Risk:   High,
            Detail: "ホストネットワーク名前空間を共有している",
            Remediation: "hostNetwork: false に設定する",
        })
    }

    // 結果出力
    printFindings(findings)
}

func isPrivileged() bool {
    // /proc/self/status の CapEff を確認
    data, err := os.ReadFile("/proc/self/status")
    if err != nil {
        return false
    }
    for _, line := range strings.Split(string(data), "\n") {
        if strings.HasPrefix(line, "CapEff:") {
            capHex := strings.TrimSpace(strings.TrimPrefix(line, "CapEff:"))
            // 全ビットが立っている = 特権
            return capHex == "000001ffffffffff" || capHex == "0000003fffffffff"
        }
    }
    return false
}

func isWritableRootFS() bool {
    testFile := "/.rootfs_write_test"
    f, err := os.Create(testFile)
    if err != nil {
        return false
    }
    f.Close()
    os.Remove(testFile)
    return true
}

func getEffectiveCaps() string {
    data, err := os.ReadFile("/proc/self/status")
    if err != nil {
        return ""
    }
    for _, line := range strings.Split(string(data), "\n") {
        if strings.HasPrefix(line, "CapEff:") {
            return strings.TrimSpace(strings.TrimPrefix(line, "CapEff:"))
        }
    }
    return ""
}

func isHostNetwork() bool {
    var stat syscall.Stat_t
    // ホストとコンテナの network namespace を比較
    err := syscall.Stat("/proc/1/ns/net", &stat)
    if err != nil {
        return false
    }
    hostIno := stat.Ino

    err = syscall.Stat("/proc/self/ns/net", &stat)
    if err != nil {
        return false
    }
    selfIno := stat.Ino

    return hostIno == selfIno
}

func printFindings(findings []Finding) {
    riskLabels := map[RiskLevel]string{
        Low: "LOW", Medium: "MEDIUM", High: "HIGH", Critical: "CRITICAL",
    }

    fmt.Println("=== Container Security Assessment ===")
    fmt.Printf("Total findings: %d\n\n", len(findings))

    for i, f := range findings {
        fmt.Printf("[%d] %s (%s)\n", i+1, f.Check, riskLabels[f.Risk])
        fmt.Printf("    Detail: %s\n", f.Detail)
        fmt.Printf("    Fix: %s\n\n", f.Remediation)
    }
}
```

---

## 2. 安全な Dockerfile

### ベストプラクティス Dockerfile (Node.js)

```dockerfile
# ---- Stage 1: ビルド ----
FROM node:20-alpine AS builder

# 非 root ユーザで作業
WORKDIR /app

# 依存関係を先にインストール (キャッシュ活用)
COPY package.json package-lock.json ./
RUN npm ci --only=production && \
    # npm キャッシュを削除してイメージサイズを削減
    npm cache clean --force

# ソースコードをコピーしてビルド
COPY . .
RUN npm run build

# ---- Stage 2: 本番イメージ ----
FROM node:20-alpine AS production

# メタデータ (OCI Image Spec)
LABEL org.opencontainers.image.title="myapp" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Example Corp"

# セキュリティアップデートを適用
RUN apk update && apk upgrade --no-cache && \
    apk add --no-cache dumb-init && \
    # SUID/SGID ビットを除去
    find / -perm /6000 -type f -exec chmod a-s {} + 2>/dev/null || true && \
    rm -rf /var/cache/apk/*

# 非 root ユーザを作成
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup -h /app -s /sbin/nologin

WORKDIR /app

# ビルド成果物のみコピー (ソースコード不要)
COPY --from=builder --chown=appuser:appgroup /app/dist ./dist
COPY --from=builder --chown=appuser:appgroup /app/node_modules ./node_modules
COPY --from=builder --chown=appuser:appgroup /app/package.json ./

# 非 root ユーザに切替
USER appuser

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/health', (r) => r.statusCode === 200 ? process.exit(0) : process.exit(1))"

# シグナル転送のため dumb-init で PID 1 問題を解決
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/server.js"]
```

### PID 1 問題の解説

```
PID 1 問題:

通常の Linux:
  PID 1 (init/systemd) → シグナルハンドリング + ゾンビ回収
    └── PID 100 (app) → SIGTERM で正常終了

コンテナ (問題あり):
  PID 1 (node server.js) → デフォルトでシグナルを無視!
    ├── SIGTERM → 無視される → docker stop が 10秒待ってSIGKILL
    └── ゾンビプロセスが回収されない

コンテナ (dumb-init で解決):
  PID 1 (dumb-init) → シグナルを適切に転送 + ゾンビ回収
    └── PID 2 (node server.js) → SIGTERM を受信して正常終了

代替策:
  - tini (Docker 公式推奨): docker run --init
  - Node.js の場合: process.on('SIGTERM', ...) を実装
```

### .dockerignore の設定

```
# .dockerignore - ビルドコンテキストから除外するファイル
.git
.gitignore
.env
.env.*
node_modules
npm-debug.log
Dockerfile
docker-compose*.yml
.dockerignore
README.md
LICENSE
.vscode
.idea
*.test.js
*.spec.js
__tests__
coverage
.github
.gitlab-ci.yml
Jenkinsfile
*.md
docs/
# シークレット関連
*.pem
*.key
*.crt
credentials.json
secrets/
```

### ベースイメージの選択

| ベースイメージ | サイズ | パッケージ数 | シェル | 脆弱性リスク | 用途 |
|--------------|--------|------------|-------|------------|------|
| ubuntu:24.04 | ~77MB | 多い (~90) | bash | 中 | 開発・デバッグ |
| debian:bookworm-slim | ~80MB | 中程度 (~70) | bash | 中 | 汎用・互換性重視 |
| alpine:3.19 | ~7MB | 最小限 (~15) | sh | 低 | 本番推奨 |
| distroless/base | ~20MB | なし (libc のみ) | なし | 最低 | 本番最推奨 |
| distroless/static | ~2MB | なし | なし | 最低 | 静的バイナリ |
| scratch | 0MB | なし | なし | なし | Go/Rust 静的バイナリ |
| chainguard/images | ~10-30MB | 最小限 | なし | 最低 | FIPS 準拠環境 |

### Alpine の musl libc 問題

```
Alpine は musl libc を使用するため、glibc 依存のバイナリで問題が発生する場合がある:

glibc (debian/ubuntu):
  - DNS 解決: /etc/nsswitch.conf, getaddrinfo() の完全な実装
  - locale: 完全なロケールサポート
  - スレッド: NPTL (成熟した実装)

musl libc (Alpine):
  - DNS 解決: /etc/resolv.conf のみ、一部の名前解決で問題
  - locale: UTF-8 のみサポート
  - スレッド: 独自実装 (一部のアプリで性能問題)
  - malloc: 大量メモリ確保時に性能劣化の可能性

対策:
  1. alpine + gcompat パッケージで glibc 互換レイヤーを追加
  2. debian-slim ベースに切り替える
  3. 本番環境で負荷テストを実施して検証する
```

### distroless イメージの活用 (Go)

```dockerfile
# Go アプリケーション用 distroless
FROM golang:1.22 AS builder

WORKDIR /app

# 依存関係を先にダウンロード
COPY go.mod go.sum ./
RUN go mod download

# ソースコードをコピーしてビルド
COPY . .

# 静的リンク + セキュリティフラグ付きビルド
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s -X main.version=$(git describe --tags)" \
    -trimpath -o /server .

# distroless nonroot イメージ
FROM gcr.io/distroless/static-debian12:nonroot

# メタデータ
LABEL org.opencontainers.image.source="https://github.com/example/myapp"

# ビルド成果物のみコピー
COPY --from=builder /server /server

# CA 証明書が必要な場合 (HTTPS 通信用)
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# nonroot ユーザで実行 (UID 65534)
USER nonroot:nonroot

ENTRYPOINT ["/server"]
```

### マルチステージビルドのパターン比較

```
パターン 1: 基本 (2 ステージ)
  builder → production
  用途: 標準的な Web アプリ

パターン 2: テスト込み (3 ステージ)
  builder → tester → production
  用途: CI/CD でテスト結果をイメージに含める

パターン 3: 開発/本番分岐
  base → dev (デバッグツール付き)
       → prod (最小イメージ)
  用途: docker compose で target を切替

パターン 4: セキュリティスキャン込み (4 ステージ)
  builder → scanner (Trivy) → signer (cosign) → production
  用途: セキュリティが最重要な環境
```

### Dockerfile のセキュリティチェックリスト

```
+-------+------------------------------------------+----------+
| 番号  | チェック項目                               | 優先度    |
+-------+------------------------------------------+----------+
| D-01  | USER で非 root を指定しているか             | 必須     |
| D-02  | マルチステージビルドを使用しているか          | 必須     |
| D-03  | 固定バージョン + ダイジェストのベースイメージか | 必須     |
| D-04  | COPY --chown で適切な所有者を設定しているか   | 必須     |
| D-05  | .dockerignore でシークレットを除外しているか  | 必須     |
| D-06  | HEALTHCHECK を設定しているか                | 推奨     |
| D-07  | LABEL でメタデータを付与しているか            | 推奨     |
| D-08  | RUN で apt/apk キャッシュを削除しているか     | 推奨     |
| D-09  | SUID/SGID ビットを除去しているか             | 推奨     |
| D-10  | ENTRYPOINT で init プロセスを使用しているか   | 推奨     |
| D-11  | ADD ではなく COPY を使用しているか            | 推奨     |
| D-12  | ENV にシークレットを設定していないか           | 必須     |
+-------+------------------------------------------+----------+
```

---

## 3. イメージスキャン

### Trivy によるスキャン

```bash
# イメージの脆弱性スキャン
trivy image --severity HIGH,CRITICAL myapp:latest

# 出力例:
# myapp:latest (alpine 3.19.0)
# ============================
# Total: 3 (HIGH: 2, CRITICAL: 1)
#
# +----------+---------------+----------+-------------------+
# | Library  | Vulnerability | Severity | Fixed Version     |
# +----------+---------------+----------+-------------------+
# | libcurl  | CVE-2024-XXX  | CRITICAL | 8.5.0-r1          |
# | openssl  | CVE-2024-YYY  | HIGH     | 3.1.4-r3          |
# +----------+---------------+----------+-------------------+

# OS パッケージ + 言語パッケージの両方をスキャン
trivy image --scanners vuln myapp:latest

# シークレット検知
trivy image --scanners secret myapp:latest

# ライセンスチェック
trivy image --scanners license --severity HIGH myapp:latest

# Dockerfile のベストプラクティスチェック
trivy config Dockerfile

# SBOM 生成 (CycloneDX 形式)
trivy image --format cyclonedx --output sbom.json myapp:latest

# SBOM 生成 (SPDX 形式)
trivy image --format spdx-json --output sbom.spdx.json myapp:latest

# CI/CD での自動ゲート (CRITICAL があれば exit code 1)
trivy image --exit-code 1 --severity CRITICAL myapp:latest

# JSON 出力 (パイプライン統合向け)
trivy image --format json --output results.json myapp:latest

# SARIF 出力 (GitHub Security タブ連携)
trivy image --format sarif --output results.sarif myapp:latest
```

### イメージスキャンツール比較

| ツール | 対象 | DB 更新 | SBOM | シークレット検知 | ライセンス | 速度 |
|--------|------|---------|------|----------------|----------|------|
| Trivy | OS+Lang+IaC | 自動 (OCI) | CycloneDX/SPDX | あり | あり | 高速 |
| Grype | OS+Lang | 自動 | SPDX (Syft) | なし | なし | 高速 |
| Snyk Container | OS+Lang | SaaS | あり | あり | あり | 中速 |
| Clair | OS のみ | 自動 | なし | なし | なし | 中速 |
| Docker Scout | OS+Lang | SaaS | あり | なし | あり | 高速 |

### Dockle によるイメージリント

```bash
# Dockle: CIS Docker Benchmark に基づくイメージ検査
dockle myapp:latest

# 検出例:
# WARN  - CIS-DI-0001: Create a user for the container
# WARN  - CIS-DI-0005: Enable Content trust for Docker
# WARN  - CIS-DI-0006: Add HEALTHCHECK instruction to the container image
# PASS  - CIS-DI-0008: Confirm safety of setuid/setgid files
# INFO  - CIS-DI-0009: Use COPY instead of ADD in Dockerfile
```

### CI/CD でのイメージスキャンパイプライン

```yaml
# GitHub Actions - 包括的なコンテナセキュリティパイプライン
name: Container Security
on:
  push:
    branches: [main]
  pull_request:
    paths: ['Dockerfile', 'src/**', 'package*.json']

jobs:
  scan:
    runs-on: ubuntu-latest
    permissions:
      security-events: write  # SARIF アップロード用
      contents: read

    steps:
      - uses: actions/checkout@v4

      # Hadolint: Dockerfile リンター
      - name: Hadolint
        uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
          failure-threshold: warning

      # イメージビルド
      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .

      # Trivy: 脆弱性スキャン
      - name: Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          severity: HIGH,CRITICAL
          exit-code: 1
          format: sarif
          output: trivy-results.sarif

      # Trivy: シークレットスキャン
      - name: Trivy secret scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          scanners: secret
          exit-code: 1

      # Dockle: イメージベストプラクティス
      - name: Dockle lint
        uses: erzz/dockle-action@v1
        with:
          image: myapp:${{ github.sha }}
          exit-code: 1
          exit-level: WARN

      # SBOM 生成
      - name: Generate SBOM
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          format: cyclonedx
          output: sbom.json

      # SARIF を GitHub Security タブにアップロード
      - name: Upload SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif

      # SBOM をアーティファクトとして保存
      - name: Upload SBOM
        uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: sbom.json
```

### SBOM (Software Bill of Materials) の重要性

```
SBOM の構造:

+--------------------------------------------------+
|  Container Image: myapp:v1.0.0                    |
|  ├── OS: Alpine Linux 3.19.0                     |
|  │   ├── musl 1.2.4-r2                          |
|  │   ├── openssl 3.1.4-r1                       |
|  │   ├── zlib 1.3-r2                            |
|  │   └── ... (15 packages)                      |
|  │                                               |
|  ├── Runtime: Node.js 20.11.0                    |
|  │                                               |
|  └── App Dependencies:                           |
|      ├── express 4.18.2                          |
|      ├── helmet 7.1.0                            |
|      ├── jsonwebtoken 9.0.2                      |
|      ├── pg 8.11.3                               |
|      └── ... (142 packages)                      |
+--------------------------------------------------+

用途:
  1. 脆弱性管理: 新規 CVE が公開されたとき影響範囲を即座に特定
  2. ライセンスコンプライアンス: GPL 等のライセンス違反を検出
  3. サプライチェーン: 依存関係の透明性を確保
  4. インシデント対応: 影響を受けるコンテナを迅速に特定

フォーマット:
  - CycloneDX (OWASP 推奨): JSON/XML
  - SPDX (Linux Foundation): JSON/RDF/Tag-Value
  - Syft JSON (Anchore 独自)
```

---

## 4. ランタイム保護

### Linux Security Modules の関係

```
アプリケーション
    |
    v
+--------------------+
| システムコール      |
| (open, read, etc.) |
+--------------------+
    |
    v
+--------------------+
| LSM フック          |  ← AppArmor / SELinux がここで判定
+--------------------+
    |
    v
+--------------------+
| seccomp-BPF        |  ← システムコール単位のフィルタリング
+--------------------+
    |
    v
+--------------------+
| Linux Kernel       |
+--------------------+

組み合わせ:
  seccomp = 「どのシステムコールを許可するか」
  AppArmor = 「どのファイル/ネットワーク操作を許可するか」
  SELinux = 「どのオブジェクトにどのアクセスを許可するか」
```

### Docker セキュリティオプション

```bash
# セキュアなコンテナ実行 (全オプション解説付き)
docker run \
  --user 1001:1001 \                    # 非 root ユーザで実行
  --read-only \                         # ファイルシステム読取専用
  --tmpfs /tmp:noexec,nosuid,size=64m \ # tmp は tmpfs (実行不可, SUID不可)
  --cap-drop ALL \                      # 全 capability を削除
  --cap-add NET_BIND_SERVICE \          # 必要な capability のみ追加
  --security-opt no-new-privileges \    # 権限昇格禁止 (SUID 無効化)
  --security-opt seccomp=default.json \ # seccomp プロファイル適用
  --security-opt apparmor=docker-default \ # AppArmor プロファイル
  --memory 512m \                       # メモリ上限
  --memory-swap 512m \                  # スワップ禁止 (メモリと同値)
  --cpus 1.0 \                          # CPU 上限
  --pids-limit 100 \                    # プロセス数制限 (fork bomb 防止)
  --network app-network \               # 専用ネットワーク
  --dns 8.8.8.8 \                       # DNS サーバ指定
  --restart on-failure:3 \              # 再起動ポリシー
  --health-cmd "curl -f http://localhost:3000/health" \
  --health-interval 30s \
  --health-retries 3 \
  myapp:v1.0.0@sha256:abc123...         # ダイジェスト固定
```

### Linux Capabilities の詳細

```
+---------------------------+-------+------------------------------------------+
| Capability                | 危険度 | 説明                                     |
+---------------------------+-------+------------------------------------------+
| CAP_SYS_ADMIN            | 最高  | mount, namespace, 多数の特権操作          |
| CAP_SYS_PTRACE           | 最高  | 他プロセスのデバッグ/メモリ読取            |
| CAP_SYS_MODULE           | 最高  | カーネルモジュールのロード                 |
| CAP_NET_ADMIN            | 高    | ネットワーク設定変更、iptables             |
| CAP_NET_RAW              | 高    | RAW ソケット (ARP スプーフィング等)        |
| CAP_DAC_OVERRIDE         | 高    | ファイル権限チェックのバイパス              |
| CAP_SETUID               | 高    | UID 変更 (権限昇格)                       |
| CAP_SETGID               | 高    | GID 変更                                 |
| CAP_CHOWN                | 中    | ファイル所有者変更                         |
| CAP_KILL                 | 中    | 任意プロセスへのシグナル送信               |
| CAP_NET_BIND_SERVICE     | 低    | 1024 未満のポートバインド                  |
| CAP_SETFCAP              | 中    | ファイル capability の設定                 |
+---------------------------+-------+------------------------------------------+

Docker デフォルト (14 capabilities):
  CAP_CHOWN, CAP_DAC_OVERRIDE, CAP_FSETID, CAP_FOWNER,
  CAP_MKNOD, CAP_NET_RAW, CAP_SETGID, CAP_SETUID,
  CAP_SETFCAP, CAP_SETPCAP, CAP_NET_BIND_SERVICE,
  CAP_SYS_CHROOT, CAP_KILL, CAP_AUDIT_WRITE

推奨: --cap-drop ALL してから必要な capability のみ --cap-add する
```

### カスタム seccomp プロファイル

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "defaultErrnoRet": 1,
  "architectures": [
    "SCMP_ARCH_X86_64",
    "SCMP_ARCH_AARCH64"
  ],
  "syscalls": [
    {
      "names": [
        "accept", "accept4", "access", "bind", "brk",
        "clone", "close", "connect", "dup", "dup2", "dup3",
        "epoll_create", "epoll_create1", "epoll_ctl", "epoll_wait",
        "execve", "exit", "exit_group",
        "fcntl", "fstat", "futex",
        "getdents64", "getpid", "getsockname", "getsockopt",
        "ioctl", "listen", "lseek",
        "mmap", "mprotect", "munmap",
        "nanosleep", "open", "openat",
        "pipe", "pipe2", "poll",
        "read", "readlink", "recvfrom", "recvmsg",
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn",
        "select", "sendmsg", "sendto", "set_tid_address",
        "setsockopt", "shutdown", "sigaltstack", "socket",
        "stat", "statfs",
        "write", "writev"
      ],
      "action": "SCMP_ACT_ALLOW"
    },
    {
      "names": [
        "ptrace", "personality", "mount", "umount2",
        "pivot_root", "kexec_load", "init_module",
        "finit_module", "delete_module", "reboot",
        "settimeofday", "stime", "clock_settime"
      ],
      "action": "SCMP_ACT_ERRNO",
      "comment": "危険なシステムコールを明示的にブロック"
    }
  ]
}
```

### Kubernetes SecurityContext

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  labels:
    app: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      # Pod レベルのセキュリティ
      securityContext:
        runAsNonRoot: true          # root 実行を禁止
        runAsUser: 1001             # 実行ユーザ
        runAsGroup: 1001            # 実行グループ
        fsGroup: 1001               # ボリュームの所有グループ
        fsGroupChangePolicy: OnRootMismatch  # 所有者変更を最小化
        seccompProfile:
          type: RuntimeDefault      # デフォルトの seccomp プロファイル
        supplementalGroups: [1001]

      containers:
        - name: myapp
          image: myapp:v1.0.0@sha256:abc123def456...  # ダイジェスト固定

          # コンテナレベルのセキュリティ
          securityContext:
            allowPrivilegeEscalation: false  # 権限昇格禁止
            readOnlyRootFilesystem: true     # 読取専用 FS
            privileged: false                # 特権モード禁止
            capabilities:
              drop: ["ALL"]                  # 全 capability を削除
              # add: ["NET_BIND_SERVICE"]    # 必要な場合のみ追加
            seccompProfile:
              type: RuntimeDefault

          # リソース制限 (必須)
          resources:
            limits:
              memory: "512Mi"
              cpu: "500m"
              ephemeral-storage: "1Gi"
            requests:
              memory: "256Mi"
              cpu: "250m"

          # ヘルスチェック
          livenessProbe:
            httpGet:
              path: /health
              port: 3000
            initialDelaySeconds: 10
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3

          readinessProbe:
            httpGet:
              path: /ready
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 10

          startupProbe:
            httpGet:
              path: /health
              port: 3000
            failureThreshold: 30
            periodSeconds: 10

          # ポート定義
          ports:
            - containerPort: 3000
              protocol: TCP

          # 環境変数 (シークレットは Secret リソースから)
          env:
            - name: NODE_ENV
              value: "production"
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: password

          # ボリュームマウント
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: app-config
              mountPath: /app/config
              readOnly: true

      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 64Mi
        - name: app-config
          configMap:
            name: myapp-config

      # イメージの pull ポリシー
      imagePullPolicy: Always

      # サービスアカウントのトークン自動マウントを無効化
      automountServiceAccountToken: false

      # Node 選択とトレランス
      nodeSelector:
        kubernetes.io/os: linux
```

### Pod Security Standards (PSS)

```
Kubernetes の Pod セキュリティの 3 段階:

+------------------+------------------------------------------+
| レベル           | 制限内容                                  |
+------------------+------------------------------------------+
| Privileged       | 制限なし (全て許可)                        |
|                  | 用途: システムコンポーネント               |
+------------------+------------------------------------------+
| Baseline         | 最小限の制限                              |
|                  | - hostNetwork/hostPID/hostIPC 禁止       |
|                  | - privileged コンテナ禁止                 |
|                  | - 危険な capabilities (SYS_ADMIN等) 禁止  |
|                  | - hostPath ボリューム禁止                 |
|                  | 用途: 一般的なワークロード                 |
+------------------+------------------------------------------+
| Restricted       | 厳格な制限 (本番推奨)                      |
|                  | - Baseline の全制限 +                    |
|                  | - runAsNonRoot 必須                      |
|                  | - allowPrivilegeEscalation: false 必須   |
|                  | - capabilities drop ALL 必須             |
|                  | - seccomp RuntimeDefault 必須            |
|                  | 用途: セキュリティ重視の本番環境           |
+------------------+------------------------------------------+
```

```yaml
# Namespace に Pod Security Standards を適用
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    # enforce: 違反する Pod の作成を拒否
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest

    # audit: 違反を監査ログに記録 (作成は許可)
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/audit-version: latest

    # warn: 違反時に警告メッセージを表示 (作成は許可)
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/warn-version: latest
```

### Kubernetes NetworkPolicy

```yaml
# Default deny (全通信をデフォルトで拒否)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}   # 全 Pod に適用
  policyTypes:
    - Ingress
    - Egress

---
# アプリケーション固有のポリシー
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: myapp-network-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
    - Ingress
    - Egress

  ingress:
    # Ingress コントローラからの HTTPS のみ受信
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
          podSelector:
            matchLabels:
              app.kubernetes.io/name: ingress-nginx
      ports:
        - port: 3000
          protocol: TCP

  egress:
    # データベースへの通信
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - port: 5432
          protocol: TCP

    # Redis への通信
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - port: 6379
          protocol: TCP

    # DNS 解決を許可
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - port: 53
          protocol: UDP
        - port: 53
          protocol: TCP

    # 外部 API への HTTPS 通信
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
            except:
              - 169.254.169.254/32  # メタデータ API をブロック
              - 10.0.0.0/8          # 内部ネットワークをブロック
      ports:
        - port: 443
          protocol: TCP
```

---

## 5. イメージの署名と検証

### コンテナサプライチェーンの全体像

```
開発者        CI/CD            レジストリ          デプロイ先
  |              |                  |                  |
  | git push     |                  |                  |
  +----------->  |                  |                  |
  |              | docker build     |                  |
  |              +-------+         |                  |
  |              |       |         |                  |
  |              | trivy scan      |                  |
  |              +-------+         |                  |
  |              |       |         |                  |
  |              | cosign sign     |                  |
  |              +-------+         |                  |
  |              |       |         |                  |
  |              | SBOM 生成       |                  |
  |              +-------+         |                  |
  |              |       | push    |                  |
  |              +-------|-------> |                  |
  |              |       |         | pull             |
  |              |       |         +----------------> |
  |              |       |         |                  |
  |              |       |         | cosign verify    |
  |              |       |         | (Admission       |
  |              |       |         |  Controller)     |
  |              |       |         +----------------> |
  |              |       |         |                  | deploy
```

### cosign によるイメージ署名

```bash
# キーペア生成
cosign generate-key-pair

# イメージに署名
cosign sign --key cosign.key myregistry/myapp:v1.0.0

# 署名の検証
cosign verify --key cosign.pub myregistry/myapp:v1.0.0

# キーレス署名 (Sigstore/Fulcio を使用)
# CI/CD の OIDC トークンを使用して署名
cosign sign --yes myregistry/myapp:v1.0.0
# → Fulcio が一時的な証明書を発行
# → Rekor に署名ログを記録 (透明性ログ)

# キーレス署名の検証
cosign verify \
  --certificate-identity "https://github.com/myorg/myapp/.github/workflows/build.yml@refs/heads/main" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  myregistry/myapp:v1.0.0

# SBOM をイメージにアタッチ
cosign attach sbom --sbom sbom.json myregistry/myapp:v1.0.0

# アテステーション (ビルド情報の証明)
cosign attest --predicate provenance.json --key cosign.key myregistry/myapp:v1.0.0

# SLSA Provenance の検証
cosign verify-attestation --key cosign.pub --type slsaprovenance myregistry/myapp:v1.0.0
```

### Kyverno によるポリシー適用

```yaml
# Kyverno: 署名済みイメージのみ許可
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: verify-image-signature
spec:
  validationFailureAction: Enforce
  background: false
  rules:
    - name: verify-cosign-signature
      match:
        any:
          - resources:
              kinds: ["Pod"]
      verifyImages:
        - imageReferences: ["myregistry/*"]
          attestors:
            - entries:
                - keys:
                    publicKeys: |-
                      -----BEGIN PUBLIC KEY-----
                      MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE...
                      -----END PUBLIC KEY-----

    - name: require-digest
      match:
        any:
          - resources:
              kinds: ["Pod"]
      validate:
        message: "イメージはダイジェストで固定する必要があります"
        pattern:
          spec:
            containers:
              - image: "*@sha256:*"

    - name: disallow-latest-tag
      match:
        any:
          - resources:
              kinds: ["Pod"]
      validate:
        message: "latest タグは使用禁止です"
        pattern:
          spec:
            containers:
              - image: "!*:latest"

---
# Kyverno: イメージレジストリの制限
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: restrict-image-registries
spec:
  validationFailureAction: Enforce
  rules:
    - name: allowed-registries
      match:
        any:
          - resources:
              kinds: ["Pod"]
      validate:
        message: "許可されたレジストリからのイメージのみ使用可能です"
        pattern:
          spec:
            containers:
              - image: "myregistry.azurecr.io/* | gcr.io/my-project/*"
```

### OPA Gatekeeper によるポリシー

```yaml
# ConstraintTemplate: 許可レジストリチェック
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8sallowedregistries
spec:
  crd:
    spec:
      names:
        kind: K8sAllowedRegistries
      validation:
        openAPIV3Schema:
          type: object
          properties:
            registries:
              type: array
              items:
                type: string
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8sallowedregistries

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not startswith(container.image, input.parameters.registries[_])
          msg := sprintf("イメージ '%v' は許可されたレジストリからのものではありません", [container.image])
        }

---
# Constraint: 上記テンプレートの適用
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sAllowedRegistries
metadata:
  name: allowed-registries
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    namespaces: ["production"]
  parameters:
    registries:
      - "myregistry.azurecr.io/"
      - "gcr.io/my-project/"
```

---

## 6. ランタイムセキュリティ監視

### Falco によるランタイム検知

```
Falco のアーキテクチャ:

+------------------+
| eBPF / kmod      |  ← カーネルレベルでシステムコールを監視
+------------------+
        |
        v
+------------------+
| Falco Engine     |  ← ルールに基づいてイベントを評価
+------------------+
        |
        v
+------------------+
| 出力先:           |
| - stdout/stderr  |
| - Syslog         |
| - HTTP (webhook) |
| - gRPC           |
| - Kafka          |
| - Slack/PagerDuty|
+------------------+
```

```yaml
# Falco カスタムルール
- rule: Shell spawned in container
  desc: コンテナ内でシェルが起動された
  condition: >
    container and
    spawned_process and
    proc.name in (bash, sh, zsh, dash, ksh) and
    not container.image.repository in (allowed_shell_images)
  output: >
    Shell spawned in container
    (user=%user.name container=%container.name
    image=%container.image.repository
    shell=%proc.name parent=%proc.pname
    cmdline=%proc.cmdline)
  priority: WARNING
  tags: [container, shell, mitre_execution]

- rule: Unexpected outbound connection
  desc: コンテナから予期しない外部通信
  condition: >
    container and
    outbound and
    not fd.sip in (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) and
    not fd.sport in (443, 80, 53) and
    not container.image.repository in (allowed_external_images)
  output: >
    Unexpected outbound connection
    (container=%container.name image=%container.image.repository
    connection=%fd.name)
  priority: NOTICE

- rule: Read sensitive file in container
  desc: コンテナ内で機密ファイルが読まれた
  condition: >
    container and
    open_read and
    fd.name in (/etc/shadow, /etc/passwd, /proc/self/environ) and
    not proc.name in (login, su, sudo)
  output: >
    Sensitive file read in container
    (user=%user.name container=%container.name
    file=%fd.name image=%container.image.repository)
  priority: WARNING

- rule: Crypto mining detected
  desc: クリプトマイニングの兆候を検知
  condition: >
    container and
    spawned_process and
    (proc.name in (xmrig, minerd, cpuminer) or
     proc.args contains "stratum+tcp" or
     proc.args contains "pool.minexmr")
  output: >
    Crypto mining process detected
    (container=%container.name image=%container.image.repository
    process=%proc.name cmdline=%proc.cmdline)
  priority: CRITICAL
```

### Falco のデプロイ (Helm)

```bash
# Helm で Falco をインストール
helm repo add falcosecurity https://falcosecurity.github.io/charts
helm repo update

helm install falco falcosecurity/falco \
  --namespace falco-system \
  --create-namespace \
  --set falcosidekick.enabled=true \
  --set falcosidekick.config.slack.webhookurl="https://hooks.slack.com/..." \
  --set falcosidekick.config.slack.minimumpriority=warning \
  --set driver.kind=ebpf \
  --set collectors.kubernetes.enabled=true
```

---

## 7. シークレット管理

### コンテナ内のシークレット管理パターン

```
アンチパターン:
  1. ENV にシークレットを埋め込む → docker inspect で露出
  2. イメージにシークレットファイルを COPY → レイヤーに残存
  3. docker-compose.yml にハードコード → Git に流出

推奨パターン:
  1. Kubernetes Secrets (Base64 エンコード、暗号化なし)
     → 最低限。etcd の encryption-at-rest を有効化必須

  2. External Secrets Operator + クラウドシークレット管理
     → AWS Secrets Manager / GCP Secret Manager / Azure Key Vault

  3. HashiCorp Vault
     → 動的シークレット生成、自動ローテーション、監査ログ

  4. Sealed Secrets (Bitnami)
     → Git にコミット可能な暗号化 Secrets
```

```yaml
# External Secrets Operator の例
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-credentials
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    kind: ClusterSecretStore
    name: aws-secrets-manager
  target:
    name: db-credentials
    creationPolicy: Owner
  data:
    - secretKey: password
      remoteRef:
        key: production/db/credentials
        property: password
    - secretKey: username
      remoteRef:
        key: production/db/credentials
        property: username
```

---

## 8. アンチパターン

### アンチパターン 1: root でコンテナを実行

```dockerfile
# NG: root で実行 (デフォルト)
FROM node:20
WORKDIR /app
COPY . .
CMD ["node", "server.js"]  # PID 1 が root (UID 0) で動作

# OK: 非 root ユーザで実行
FROM node:20-alpine
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup -s /sbin/nologin
WORKDIR /app
COPY --chown=appuser:appgroup . .
USER appuser
CMD ["dumb-init", "node", "server.js"]
```

**影響**: コンテナエスケープ脆弱性 (例: CVE-2019-5736) が悪用された場合、ホスト OS の root 権限を取得される。root であれば /proc, /sys への書き込みなどホスト侵害の可能性が大幅に上がる。

**検出**: `docker inspect --format '{{.Config.User}}' <container>` が空または "0" なら root 実行。

### アンチパターン 2: latest タグの使用

```dockerfile
# NG: latest タグ (内容が変わりうる)
FROM node:latest
# → ビルドのたびに異なるバージョンが使われる可能性
# → サプライチェーン攻撃でタグが上書きされるリスク

# NG: バージョンのみ (ダイジェストなし)
FROM node:20
# → パッチバージョンが変わる可能性

# OK: 固定バージョン + ダイジェスト
FROM node:20.11.0-alpine@sha256:abc123def456...
# → 完全に再現可能なビルド
# → タグが上書きされてもダイジェストで検証
```

**影響**: サプライチェーン攻撃でタグが悪意あるイメージに上書きされた場合、検知できずにデプロイされる。再現性のないビルドは監査・インシデント対応が困難になる。

### アンチパターン 3: Docker Socket のマウント

```yaml
# NG: Docker Socket をコンテナにマウント
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
# → コンテナから docker コマンドでホスト上に特権コンテナを起動可能
# → 実質的にホストの root 権限と同等

# OK: Docker-in-Docker が必要な場合
# 方法1: Kaniko (ビルド専用、デーモンレス)
# 方法2: Buildah (rootless, daemonless)
# 方法3: Docker-in-Docker (--privileged が必要で推奨しないが、
#         CI/CD で隔離された環境でのみ許容)
```

**影響**: コンテナが侵害された場合、Docker API を通じてホスト上の全コンテナの操作、新規特権コンテナの起動、ホストファイルシステムへのアクセスが可能になる。

### アンチパターン 4: シークレットの Dockerfile への埋め込み

```dockerfile
# NG: ENV にシークレットを設定
ENV DATABASE_URL="postgres://user:password@db:5432/mydb"
ENV API_KEY="sk-1234567890abcdef"
# → docker history でレイヤーから復元可能
# → docker inspect で環境変数が見える

# NG: COPY でシークレットファイルをコピー
COPY credentials.json /app/credentials.json
# → マルチステージビルドで削除しても前のレイヤーに残存

# OK: BuildKit の --mount=type=secret を使用
# syntax=docker/dockerfile:1
RUN --mount=type=secret,id=db_password \
    DB_PASS=$(cat /run/secrets/db_password) && \
    ./setup-db.sh "$DB_PASS"
# → シークレットはイメージレイヤーに保存されない

# ビルド時:
# docker build --secret id=db_password,src=./db_password.txt .
```

**影響**: イメージレジストリにプッシュされたシークレットは、イメージにアクセスできる全員に露出する。レイヤーは永続的に保存されるため、後から削除しても過去のレイヤーから復元可能。

---

## 9. エッジケース

### エッジケース 1: Alpine の DNS 解決問題

Alpine の musl libc は DNS 解決で `search` ドメインと `ndots` の処理が glibc と異なる。Kubernetes では Pod の `/etc/resolv.conf` に `ndots:5` が設定されているため、短い名前の解決で意図しない DNS クエリが大量に発生する場合がある。

```yaml
# 対策: dnsConfig で ndots を調整
spec:
  dnsConfig:
    options:
      - name: ndots
        value: "2"  # デフォルトの 5 から削減
      - name: single-request-reopen  # musl の DNS 問題回避
```

### エッジケース 2: tmpfs と noexec の問題

一部のアプリケーション (Java の JIT コンパイラなど) は `/tmp` に実行可能ファイルを生成する必要がある。`readOnlyRootFilesystem: true` + `tmpfs noexec` の組み合わせでアプリケーションが起動しない場合がある。

```yaml
# 対策: exec が必要な tmpfs は明示的に許可
volumes:
  - name: tmp-exec
    emptyDir:
      medium: Memory
      sizeLimit: 128Mi
volumeMounts:
  - name: tmp-exec
    mountPath: /tmp
    # noexec は設定しない (JIT が必要)
```

### エッジケース 3: User Namespace Remapping

Docker の User Namespace Remapping を有効にすると、コンテナ内の root (UID 0) がホスト上では非特権ユーザ (例: UID 100000) にマッピングされる。これによりコンテナエスケープ時の影響を大幅に軽減できるが、ボリュームの権限問題が発生する。

```bash
# Docker の userns-remap を有効化
# /etc/docker/daemon.json
{
  "userns-remap": "default"
}
# → /etc/subuid, /etc/subgid にマッピングが設定される
# → 既存のボリュームの権限を再設定する必要がある
```

### エッジケース 4: イメージのマルチアーキテクチャと署名

マルチアーキテクチャイメージ (manifest list) では、アーキテクチャごとに異なるダイジェストが存在する。cosign の署名はマニフェストリストのダイジェストに対して行う必要がある。

```bash
# マルチアーキテクチャイメージの構造
# docker manifest inspect myapp:v1.0.0
{
  "manifests": [
    {"platform": {"architecture": "amd64"}, "digest": "sha256:aaa..."},
    {"platform": {"architecture": "arm64"}, "digest": "sha256:bbb..."}
  ]
}
# → cosign sign は manifest list のダイジェストに対して実行
# → 各アーキテクチャのイメージが改竄されると署名検証が失敗
```

---

## 10. パフォーマンスとセキュリティのトレードオフ

### セキュリティオプションのオーバーヘッド

| オプション | CPU オーバーヘッド | メモリ影響 | ネットワーク影響 | 推奨度 |
|-----------|------------------|----------|---------------|-------|
| 非 root 実行 | なし | なし | なし | 必須 |
| read-only FS | なし (わずかに改善) | なし | なし | 必須 |
| cap-drop ALL | なし | なし | なし | 必須 |
| seccomp (default) | ~1-3% | なし | なし | 必須 |
| AppArmor | ~1-2% | なし | なし | 推奨 |
| SELinux | ~2-5% | なし | なし | 推奨 |
| User Namespace | ~1% | なし | なし | 推奨 |
| NetworkPolicy | なし | なし | ~1ms latency | 必須 |
| イメージスキャン (CI) | ビルド時 30-60秒 | N/A | N/A | 必須 |
| Falco (eBPF) | ~2-5% | ~100-200MB | なし | 推奨 |
| gVisor | ~10-30% | ~50-100MB/container | ~5-10% | 高セキュリティ環境 |
| Kata Containers | ~5-15% | ~100-200MB/container | ~3-5% | マルチテナント |

### イメージサイズとセキュリティの関係

```
イメージサイズ vs 攻撃面:

ubuntu:24.04   ████████████████████████████████████████  ~77MB  CVE ~40-60
debian-slim    ███████████████████████████████████████   ~80MB  CVE ~30-50
alpine:3.19    ████                                      ~7MB   CVE ~5-15
distroless     ██                                        ~20MB  CVE ~0-5
scratch        |                                         ~0MB   CVE 0

結論: イメージサイズが小さい ≒ パッケージ数が少ない ≒ CVE が少ない
     ただし、alpine の musl libc 固有の問題に注意
```

---

## 11. 演習問題

### 演習 1: セキュアな Dockerfile の作成 (初級)

以下の脆弱な Dockerfile を修正し、セキュリティベストプラクティスに準拠させなさい。

```dockerfile
# 脆弱な Dockerfile
FROM python:latest
WORKDIR /app
ENV DATABASE_URL=postgres://admin:password@db:5432/mydb
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD python app.py
```

**要件**:
1. 非 root ユーザで実行する
2. マルチステージビルドを使用する
3. シークレットをイメージに含めない
4. ベースイメージのバージョンを固定する
5. read-only ファイルシステムで動作するようにする

**ヒント**: `python:3.12-slim` をベースに、`--mount=type=secret` を活用する。

### 演習 2: Kubernetes SecurityContext の設計 (中級)

以下の要件を満たす Deployment マニフェストを作成しなさい。

**要件**:
1. Pod Security Standards の `restricted` レベルに準拠する
2. 外部 API (api.example.com:443) と PostgreSQL (postgres:5432) への通信のみ許可する
3. NetworkPolicy で DNS 以外の不要な通信を遮断する
4. リソース制限を適切に設定する
5. liveness/readiness/startup probe を全て設定する

**追加課題**: OPA Gatekeeper または Kyverno で、この要件をクラスタ全体に強制するポリシーを作成しなさい。

### 演習 3: コンテナサプライチェーンの構築 (上級)

以下の要件を満たす CI/CD パイプラインを構築しなさい。

**要件**:
1. Hadolint で Dockerfile をリントする
2. Trivy でイメージスキャン (CRITICAL で失敗)
3. Dockle で CIS Docker Benchmark を検証する
4. SBOM を CycloneDX 形式で生成する
5. cosign でキーレス署名する
6. Kyverno で署名検証を Kubernetes にデプロイ時に強制する

**追加課題**: SLSA Level 3 の Provenance を生成し、cosign attest でイメージにアタッチしなさい。

---

## 12. トラブルシューティング

### よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| `readOnlyRootFilesystem` で起動失敗 | アプリが `/tmp` や `/var` に書き込む | tmpfs/emptyDir をマウント |
| `runAsNonRoot` でイメージが起動しない | イメージに USER が設定されていない | Dockerfile に USER を追加 |
| `cap-drop ALL` でネットワーク接続失敗 | `NET_RAW` が必要 (ping 等) | `cap-add NET_RAW` (本当に必要か再検討) |
| Alpine で DNS 解決が遅い | musl の DNS 実装 + ndots:5 | dnsConfig で ndots を調整 |
| cosign verify が失敗 | 署名時と異なるダイジェスト | manifest list vs manifest の違いを確認 |
| NetworkPolicy が効かない | CNI が NetworkPolicy 未対応 | Calico/Cilium に変更 |
| Trivy で false positive | 未使用ライブラリの検出 | `.trivyignore` で抑制 + 根拠を記録 |
| SecurityContext の Deprecated 警告 | PSP (PodSecurityPolicy) を使用中 | PSS (Pod Security Standards) に移行 |
| イメージ pull に失敗 | ImagePullPolicy: Always + レジストリ障害 | IfNotPresent + ダイジェスト固定 |
| ゾンビプロセスの蓄積 | PID 1 が子プロセスを回収しない | dumb-init/tini を使用 |

---

## 13. FAQ

### Q1. distroless と Alpine のどちらを選ぶべきか?

シェルやデバッグツールが不要な本番環境では distroless が最もセキュアである。Alpine はシェルが含まれるためデバッグが容易で、バランスの取れた選択肢である。開発段階では Alpine を使い、本番では distroless に切り替える戦略が効果的である。ただし、distroless ではコンテナにシェルで入れないため、`kubectl debug` でエフェメラルコンテナを使うか、ログ出力を充実させて可観測性を確保する必要がある。

### Q2. コンテナの脆弱性スキャンはいつ行うべきか?

CI/CD パイプラインでのビルド時スキャン (ゲート)、レジストリでの定期スキャン (日次)、ランタイムでの継続的スキャンの 3 段階で行うのが理想的である。ビルド時に CRITICAL を見逃さず、定期スキャンで新規 CVE をキャッチする。また、ベースイメージの更新を自動化するツール (Renovate, Dependabot) を併用して、既知の脆弱性を含むイメージが長期間使われないようにする。

### Q3. read-only ファイルシステムで一時ファイルが必要な場合は?

`tmpfs` をマウントして `/tmp` を提供する。`emptyDir` (Kubernetes) や `--tmpfs` (Docker) を使い、サイズ制限と `noexec` オプションを設定する。ログは stdout/stderr に出力するか、外部ログ収集に委譲する。Java アプリケーションの場合、JIT コンパイラが `/tmp` に実行可能ファイルを生成するため `noexec` を外す必要がある場合がある。

### Q4. gVisor と Kata Containers のどちらを選ぶべきか?

gVisor はユーザ空間カーネル (Sentry) でシステムコールを仲介し、カーネル攻撃面を大幅に削減する。オーバーヘッドは大きいが (10-30%)、追加のハードウェアは不要である。Kata Containers は軽量 VM でコンテナを実行し、ハードウェアレベルの分離を提供する。ネストされた仮想化が必要なため、ベアメタルまたは対応するクラウド環境が必要である。マルチテナント環境では Kata、信頼できないコード実行では gVisor が適している。

### Q5. Kubernetes で ServiceAccount の権限を最小化するには?

`automountServiceAccountToken: false` を設定し、必要な Pod にのみ最小権限の ServiceAccount を作成する。RBAC で Pod が必要とする API リソースのみに限定した Role/ClusterRole を紐づける。Kubernetes 1.24 以降では ServiceAccount トークンはバウンドトークン (有効期限付き、オーディエンス付き) がデフォルトであるが、レガシートークンが無効化されていることを確認する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| コンテナの分離 | カーネル共有のため VM より弱い。namespaces + cgroups + LSM で強化 |
| ベースイメージ | Alpine / distroless で攻撃面を最小化。バージョン+ダイジェスト固定 |
| マルチステージビルド | ビルドツールを本番イメージに含めない。SUID/SGID を除去 |
| イメージスキャン | Trivy で HIGH/CRITICAL をゲート。SBOM を生成して管理 |
| 非 root 実行 | USER 指定 + allowPrivilegeEscalation: false + no-new-privileges |
| 読取専用 FS | readOnlyRootFilesystem: true + tmpfs で書き込み領域を限定 |
| capability 削減 | cap-drop ALL + 必要なもののみ cap-add |
| ネットワーク制限 | NetworkPolicy で default deny + 必要な通信のみ許可 |
| イメージ署名 | cosign + Kyverno/Gatekeeper で署名検証を強制 |
| ランタイム監視 | Falco (eBPF) で異常なシステムコール・ネットワーク通信を検知 |
| シークレット管理 | External Secrets Operator + クラウドシークレット管理 |
| Pod Security | Pod Security Standards (restricted) を本番 Namespace に適用 |

---

## 次に読むべきガイド

- [SAST/DAST](./03-sast-dast.md) -- コードとアプリケーションの脆弱性スキャン
- [IaCセキュリティ](../05-cloud-security/02-infrastructure-as-code-security.md) -- Kubernetes マニフェストのセキュリティチェック
- [依存関係セキュリティ](./01-dependency-security.md) -- コンテナ内の依存関係管理
- [セキュアコーディング](./00-secure-coding.md) -- コードレベルでの攻撃防御

---

## 参考文献

1. **CIS Docker Benchmark** -- https://www.cisecurity.org/benchmark/docker
2. **NIST SP 800-190 -- Application Container Security Guide** -- https://csrc.nist.gov/publications/detail/sp/800-190/final
3. **Kubernetes Security Best Practices** -- https://kubernetes.io/docs/concepts/security/
4. **Google Distroless Images** -- https://github.com/GoogleContainerTools/distroless
5. **Sigstore/cosign Documentation** -- https://docs.sigstore.dev/
6. **Falco Documentation** -- https://falco.org/docs/
7. **SLSA (Supply-chain Levels for Software Artifacts)** -- https://slsa.dev/
8. **CIS Kubernetes Benchmark** -- https://www.cisecurity.org/benchmark/kubernetes
9. **OWASP Docker Security Cheat Sheet** -- https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html

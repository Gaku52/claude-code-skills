# コンテナセキュリティ

> コンテナイメージのスキャン、最小権限でのランタイム保護、安全な Dockerfile の書き方まで、コンテナ化されたアプリケーションを守るための包括的ガイド

## この章で学ぶこと

1. **イメージセキュリティ** — ベースイメージの選択、脆弱性スキャン、マルチステージビルド
2. **ランタイム保護** — seccomp、AppArmor、非 root 実行、リードオンリーファイルシステム
3. **オーケストレーションセキュリティ** — Kubernetes のセキュリティコンテキスト、ネットワークポリシー

---

## 1. コンテナの脅威モデル

### 攻撃面の分類

```
+----------------------------------------------------------+
|                コンテナの攻撃面                              |
|----------------------------------------------------------|
|                                                          |
|  [イメージ層]                                              |
|  +-- ベースイメージの脆弱性                                |
|  +-- アプリ依存ライブラリの脆弱性                           |
|  +-- シークレットの埋め込み                                |
|  +-- 不要パッケージの含有                                  |
|                                                          |
|  [ビルド層]                                                |
|  +-- 信頼できないレジストリからの pull                      |
|  +-- タグの可変性 (latest の上書き)                        |
|  +-- CI/CD パイプラインの侵害                              |
|                                                          |
|  [ランタイム層]                                            |
|  +-- root 実行による権限昇格                               |
|  +-- コンテナエスケープ                                    |
|  +-- 過剰な Linux capabilities                            |
|  +-- ホストパスのマウント                                  |
|                                                          |
|  [ネットワーク層]                                          |
|  +-- コンテナ間の無制限通信                                |
|  +-- メタデータ API への不正アクセス                        |
+----------------------------------------------------------+
```

---

## 2. 安全な Dockerfile

### ベストプラクティス Dockerfile

```dockerfile
# ---- Stage 1: ビルド ----
FROM node:20-alpine AS builder

# 非 root ユーザで作業
WORKDIR /app

# 依存関係を先にインストール (キャッシュ活用)
COPY package.json package-lock.json ./
RUN npm ci --only=production

# ソースコードをコピーしてビルド
COPY . .
RUN npm run build

# ---- Stage 2: 本番イメージ ----
FROM node:20-alpine AS production

# セキュリティアップデートを適用
RUN apk update && apk upgrade --no-cache && \
    apk add --no-cache dumb-init && \
    rm -rf /var/cache/apk/*

# 非 root ユーザを作成
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup

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

# dumb-init で PID 1 問題を解決
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/server.js"]
```

### ベースイメージの選択

| ベースイメージ | サイズ | パッケージ数 | 脆弱性リスク | 用途 |
|--------------|--------|------------|------------|------|
| ubuntu:24.04 | ~77MB | 多い | 中 | 開発・デバッグ |
| debian:bookworm-slim | ~80MB | 中程度 | 中 | 汎用 |
| alpine:3.19 | ~7MB | 最小限 | 低 | 本番推奨 |
| distroless | ~2-20MB | なし | 最低 | 本番最推奨 |
| scratch | 0MB | なし | なし | 静的バイナリ |

### distroless イメージの活用

```dockerfile
# Go アプリケーション用 distroless
FROM golang:1.22 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /server .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=builder /server /server
USER nonroot:nonroot
ENTRYPOINT ["/server"]
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

# シークレット検知
trivy image --scanners secret myapp:latest

# Dockerfile のベストプラクティスチェック
trivy config Dockerfile

# CI/CD での自動ゲート
trivy image --exit-code 1 --severity CRITICAL myapp:latest
```

### CI/CD でのイメージスキャンパイプライン

```yaml
# GitHub Actions
name: Container Security
on:
  push:
    branches: [main]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          severity: HIGH,CRITICAL
          exit-code: 1
          format: sarif
          output: trivy-results.sarif

      - name: Upload scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif

      - name: Dockle lint
        uses: erzz/dockle-action@v1
        with:
          image: myapp:${{ github.sha }}
          exit-code: 1
```

---

## 4. ランタイム保護

### Docker セキュリティオプション

```bash
# セキュアなコンテナ実行
docker run \
  --user 1001:1001 \                    # 非 root
  --read-only \                         # ファイルシステム読取専用
  --tmpfs /tmp:noexec,nosuid,size=64m \ # tmp は tmpfs
  --cap-drop ALL \                      # 全 capability を削除
  --cap-add NET_BIND_SERVICE \          # 必要な capability のみ追加
  --security-opt no-new-privileges \    # 権限昇格禁止
  --security-opt seccomp=default.json \ # seccomp プロファイル
  --memory 512m \                       # メモリ制限
  --cpus 1.0 \                          # CPU 制限
  --pids-limit 100 \                    # プロセス数制限
  --network app-network \               # 専用ネットワーク
  myapp:latest
```

### Kubernetes SecurityContext

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  template:
    spec:
      # Pod レベルのセキュリティ
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        runAsGroup: 1001
        fsGroup: 1001
        seccompProfile:
          type: RuntimeDefault

      containers:
        - name: myapp
          image: myapp:v1.0.0@sha256:abc123...  # ダイジェスト固定
          # コンテナレベルのセキュリティ
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          resources:
            limits:
              memory: "512Mi"
              cpu: "500m"
            requests:
              memory: "256Mi"
              cpu: "250m"
          volumeMounts:
            - name: tmp
              mountPath: /tmp
          livenessProbe:
            httpGet:
              path: /health
              port: 3000
            initialDelaySeconds: 10
            periodSeconds: 30

      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 64Mi

      # イメージの pull ポリシー
      imagePullPolicy: Always

      # サービスアカウントのトークン自動マウントを無効化
      automountServiceAccountToken: false
```

### Kubernetes NetworkPolicy

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: myapp-network-policy
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Ingress コントローラからのみ受信
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - port: 3000
  egress:
    # データベースへの通信のみ許可
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - port: 5432
    # DNS 解決を許可
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - port: 53
          protocol: UDP
```

---

## 5. イメージの署名と検証

```bash
# cosign でイメージに署名
cosign sign --key cosign.key myregistry/myapp:v1.0.0

# 署名の検証
cosign verify --key cosign.pub myregistry/myapp:v1.0.0

# Kubernetes で署名検証を強制 (Sigstore Policy Controller)
# または Kyverno ポリシー
```

```yaml
# Kyverno: 署名済みイメージのみ許可
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: verify-image-signature
spec:
  validationFailureAction: Enforce
  rules:
    - name: verify-signature
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
```

---

## 6. アンチパターン

### アンチパターン 1: root でコンテナを実行

```dockerfile
# NG: root で実行 (デフォルト)
FROM node:20
WORKDIR /app
COPY . .
CMD ["node", "server.js"]  # PID 1 が root で動作

# OK: 非 root ユーザで実行
FROM node:20-alpine
RUN adduser -D appuser
WORKDIR /app
COPY --chown=appuser . .
USER appuser
CMD ["node", "server.js"]
```

**影響**: コンテナエスケープ脆弱性が悪用された場合、ホスト OS の root 権限を取得される。

### アンチパターン 2: latest タグの使用

```dockerfile
# NG: latest タグ (内容が変わりうる)
FROM node:latest
# → ビルドのたびに異なるバージョンが使われる可能性

# OK: 固定バージョン + ダイジェスト
FROM node:20.11.0-alpine@sha256:abc123def456...
# → 完全に再現可能なビルド
```

**影響**: サプライチェーン攻撃でタグが上書きされた場合、悪意あるイメージが使用される。

---

## 7. FAQ

### Q1. distroless と Alpine のどちらを選ぶべきか?

シェルやデバッグツールが不要な本番環境では distroless が最もセキュアである。Alpine はシェルが含まれるためデバッグが容易で、バランスの取れた選択肢である。開発段階では Alpine を使い、本番では distroless に切り替える戦略が効果的である。

### Q2. コンテナの脆弱性スキャンはいつ行うべきか?

CI/CD パイプラインでのビルド時スキャン (ゲート)、レジストリでの定期スキャン (日次)、ランタイムでの継続的スキャンの 3 段階で行うのが理想的である。ビルド時に CRITICAL を見逃さず、定期スキャンで新規 CVE をキャッチする。

### Q3. read-only ファイルシステムで一時ファイルが必要な場合は?

`tmpfs` をマウントして `/tmp` を提供する。`emptyDir` (Kubernetes) や `--tmpfs` (Docker) を使い、サイズ制限と `noexec` オプションを設定する。ログは stdout/stderr に出力するか、外部ログ収集に委譲する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| ベースイメージ | Alpine / distroless で攻撃面を最小化 |
| マルチステージビルド | ビルドツールを本番イメージに含めない |
| イメージスキャン | Trivy で HIGH/CRITICAL をゲート |
| 非 root 実行 | USER 指定 + allowPrivilegeEscalation: false |
| 読取専用FS | readOnlyRootFilesystem: true + tmpfs |
| capability 削減 | cap-drop ALL + 必要なもののみ cap-add |
| ネットワーク制限 | NetworkPolicy で通信を最小限に |
| イメージ署名 | cosign + ポリシーで署名検証を強制 |

---

## 次に読むべきガイド

- [SAST/DAST](./03-sast-dast.md) — コードとアプリケーションの脆弱性スキャン
- [IaCセキュリティ](../05-cloud-security/02-infrastructure-as-code-security.md) — Kubernetes マニフェストのセキュリティチェック
- [依存関係セキュリティ](./01-dependency-security.md) — コンテナ内の依存関係管理

---

## 参考文献

1. **CIS Docker Benchmark** — https://www.cisecurity.org/benchmark/docker
2. **NIST SP 800-190 — Application Container Security Guide** — https://csrc.nist.gov/publications/detail/sp/800-190/final
3. **Kubernetes Security Best Practices** — https://kubernetes.io/docs/concepts/security/
4. **Google Distroless Images** — https://github.com/GoogleContainerTools/distroless

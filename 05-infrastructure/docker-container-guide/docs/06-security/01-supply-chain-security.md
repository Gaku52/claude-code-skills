# サプライチェーンセキュリティ (Supply Chain Security)

> コンテナイメージの署名 (cosign)、SBOM (Software Bill of Materials) の生成・検証を通じて、ビルドからデプロイまでのソフトウェアサプライチェーン全体の信頼性を確保する手法を学ぶ。

## この章で学ぶこと

1. **イメージ署名と検証 (cosign / Sigstore)** -- コンテナイメージにデジタル署名を付与し、改ざんされていないことを検証する仕組みを構築する
2. **SBOM の生成と活用** -- ソフトウェアの構成要素を一覧化し、脆弱性の追跡と規制対応を実現する
3. **CI/CD パイプラインでのサプライチェーン保護** -- ビルドの来歴 (Provenance) 記録と、ポリシーベースのデプロイ制御を実装する

---

## 1. サプライチェーンセキュリティの全体像

```
+------------------------------------------------------------------+
|              ソフトウェアサプライチェーンの脅威と対策                  |
+------------------------------------------------------------------+
|                                                                  |
|  [ソースコード]                                                   |
|    脅威: 依存パッケージの改ざん、typosquatting                     |
|    対策: 依存関係の固定 (lockfile)、npm audit                     |
|       |                                                          |
|  [ビルド]                                                        |
|    脅威: ビルド環境の侵害、不正なビルドステップ                     |
|    対策: 再現可能ビルド、ビルド来歴 (Provenance)                   |
|       |                                                          |
|  [イメージ]                                                      |
|    脅威: イメージの改ざん、レジストリ侵害                          |
|    対策: イメージ署名 (cosign)、SBOM、脆弱性スキャン               |
|       |                                                          |
|  [レジストリ]                                                    |
|    脅威: 不正イメージの push、タグの上書き                         |
|    対策: イミュータブルタグ、署名検証ポリシー                      |
|       |                                                          |
|  [デプロイ]                                                      |
|    脅威: 未署名イメージのデプロイ                                  |
|    対策: Admission Controller (署名検証ゲート)                    |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 2. cosign によるイメージ署名

### 2.1 cosign の概要

```
+------------------------------------------------------------------+
|              cosign 署名フロー                                     |
+------------------------------------------------------------------+
|                                                                  |
|  [ビルド]                                                        |
|    docker build -t ghcr.io/myorg/myapp:1.0.0 .                  |
|    docker push ghcr.io/myorg/myapp:1.0.0                        |
|       |                                                          |
|  [署名] (CI/CD)                                                  |
|    cosign sign ghcr.io/myorg/myapp@sha256:abc123...              |
|       |                                                          |
|       +-- Keyless (推奨): OIDC トークンで一時鍵を生成             |
|       |     GitHub Actions → Fulcio → Rekor (透明性ログ)         |
|       |                                                          |
|       +-- Key-pair: 事前生成した鍵ペアで署名                      |
|       |     cosign.key (秘密鍵) / cosign.pub (公開鍵)             |
|       |                                                          |
|  [検証] (デプロイ時)                                              |
|    cosign verify ghcr.io/myorg/myapp@sha256:abc123...            |
|       |                                                          |
|       +-- Keyless: Fulcio の証明書 + Rekor ログで検証             |
|       +-- Key-pair: 公開鍵で検証                                  |
|                                                                  |
+------------------------------------------------------------------+
```

### 2.2 cosign のインストールと鍵生成

```bash
# インストール
# macOS
brew install cosign

# Linux
curl -L https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64 \
  -o /usr/local/bin/cosign && chmod +x /usr/local/bin/cosign

# 鍵ペア生成 (Key-pair 方式)
cosign generate-key-pair
# → cosign.key (秘密鍵、パスワード保護)
# → cosign.pub (公開鍵、配布用)

# KMS を使う場合
cosign generate-key-pair --kms awskms:///alias/cosign-key
```

### 2.3 イメージ署名と検証

```bash
# イメージのビルドとプッシュ
docker build -t ghcr.io/myorg/myapp:1.0.0 .
docker push ghcr.io/myorg/myapp:1.0.0

# ダイジェストを取得 (タグではなくダイジェストで署名)
DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/myorg/myapp:1.0.0)

# Key-pair 方式で署名
cosign sign --key cosign.key $DIGEST

# Keyless 方式で署名 (推奨、CI/CD 向け)
COSIGN_EXPERIMENTAL=1 cosign sign $DIGEST

# 検証 (Key-pair)
cosign verify --key cosign.pub ghcr.io/myorg/myapp:1.0.0

# 検証 (Keyless)
cosign verify \
  --certificate-identity="https://github.com/myorg/myapp/.github/workflows/build.yml@refs/tags/v1.0.0" \
  --certificate-oidc-issuer="https://token.actions.githubusercontent.com" \
  ghcr.io/myorg/myapp:1.0.0
```

### 2.4 GitHub Actions での Keyless 署名

```yaml
# .github/workflows/build-sign.yml
name: Build and Sign

on:
  push:
    tags: ['v*']

permissions:
  contents: read
  packages: write
  id-token: write    # Keyless 署名に必要

jobs:
  build-and-sign:
    runs-on: ubuntu-latest
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
            type=sha

      - uses: docker/build-push-action@v5
        id: build
        with:
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

      - uses: sigstore/cosign-installer@v3

      # Keyless 署名 (OIDC トークンベース)
      - name: Sign image
        run: |
          cosign sign --yes \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}

      # SBOM を添付
      - name: Attach SBOM
        run: |
          trivy image --format spdx-json \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }} \
            > sbom.spdx.json
          cosign attach sbom \
            --sbom sbom.spdx.json \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}
```

---

## 3. SBOM (Software Bill of Materials)

### 3.1 SBOM とは

```
+------------------------------------------------------------------+
|              SBOM の構造                                          |
+------------------------------------------------------------------+
|                                                                  |
|  SBOM (myapp:1.0.0)                                             |
|  |                                                               |
|  +-- OS パッケージ                                               |
|  |     +-- alpine-baselayout 3.4.3                               |
|  |     +-- musl 1.2.4                                            |
|  |     +-- openssl 3.1.4 ← CVE-2024-XXXX (修正済み)             |
|  |     +-- ...                                                   |
|  |                                                               |
|  +-- アプリ依存関係                                               |
|  |     +-- express 4.18.2                                        |
|  |     +-- pg 8.11.3                                             |
|  |     +-- @prisma/client 5.7.1                                  |
|  |     +-- ...                                                   |
|  |                                                               |
|  +-- メタデータ                                                   |
|       +-- ビルド日時: 2025-01-15T10:30:00Z                       |
|       +-- ビルドツール: docker buildx 0.12.0                     |
|       +-- ソース: github.com/myorg/myapp@abc1234                 |
|                                                                  |
+------------------------------------------------------------------+
```

### 3.2 SBOM 生成ツール

```bash
# Trivy で SBOM 生成 (SPDX 形式)
trivy image --format spdx-json --output sbom-spdx.json myapp:latest

# Trivy で SBOM 生成 (CycloneDX 形式)
trivy image --format cyclonedx --output sbom-cdx.json myapp:latest

# Syft で SBOM 生成 (Anchore 製)
syft myapp:latest -o spdx-json > sbom-syft.json

# Docker Scout (Docker Desktop 統合)
docker scout sbom myapp:latest

# BuildKit で SBOM を自動生成
docker buildx build --sbom=true -t myapp:latest .
```

### 3.3 SBOM 形式の比較

| 項目 | SPDX | CycloneDX | SWID |
|------|------|-----------|------|
| 標準化団体 | Linux Foundation | OWASP | ISO/IEC |
| フォーマット | JSON, RDF, Tag-Value | JSON, XML, Protobuf | XML |
| ライセンス情報 | 非常に詳細 | 基本的 | 基本的 |
| 脆弱性情報 | 外部連携 | VEX 統合 | なし |
| 採用事例 | 米国政府 (EO 14028) | セキュリティツール | レガシー |
| ツール対応 | 広い | 広い | 限定的 |
| 推奨用途 | ライセンスコンプライアンス | セキュリティ分析 | 資産管理 |

### 3.4 SBOM からの脆弱性スキャン

```bash
# SBOM を入力として脆弱性スキャン
trivy sbom sbom-spdx.json

# 特定の CVE を含むかチェック
trivy sbom sbom-spdx.json | grep CVE-2024-XXXX

# grype で SBOM スキャン
grype sbom:sbom-cdx.json
```

---

## 4. ビルド来歴 (Provenance)

### 4.1 SLSA (Supply chain Levels for Software Artifacts)

```
+------------------------------------------------------------------+
|              SLSA レベル                                          |
+------------------------------------------------------------------+
|                                                                  |
|  Level 0: なし                                                   |
|    → 何もしない                                                  |
|                                                                  |
|  Level 1: 来歴の存在                                             |
|    → ビルドプロセスの記録が存在する                                |
|    → 手動ビルドでも可                                             |
|                                                                  |
|  Level 2: ホスティングされたビルド                                 |
|    → CI/CD サービスでビルド                                       |
|    → 来歴が自動生成される                                         |
|                                                                  |
|  Level 3: ハード化されたビルド                                     |
|    → ビルド環境が改ざん耐性を持つ                                  |
|    → 来歴が暗号署名される                                         |
|    → GitHub Actions + SLSA Generator が対応                      |
|                                                                  |
+------------------------------------------------------------------+
```

### 4.2 GitHub Actions での Provenance 生成

```yaml
# .github/workflows/slsa-build.yml
name: SLSA Build

on:
  push:
    tags: ['v*']

permissions:
  contents: read
  packages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      # BuildKit の Provenance 生成を有効化
      - uses: docker/build-push-action@v5
        id: build
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}
          provenance: true     # SLSA Provenance を自動生成
          sbom: true           # SBOM を自動生成

      - uses: sigstore/cosign-installer@v3

      - name: Sign with cosign
        run: |
          cosign sign --yes \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}

      # Provenance の検証
      - name: Verify provenance
        run: |
          cosign verify-attestation \
            --type slsaprovenance \
            --certificate-identity-regexp="https://github.com/${{ github.repository }}/" \
            --certificate-oidc-issuer="https://token.actions.githubusercontent.com" \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}
```

---

## 5. デプロイ時のポリシー適用

### 5.1 Kubernetes Admission Controller

```yaml
# Kyverno ポリシー: 署名済みイメージのみ許可
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
              kinds:
                - Pod
      verifyImages:
        - imageReferences:
            - "ghcr.io/myorg/*"
          attestors:
            - entries:
                - keyless:
                    subject: "https://github.com/myorg/*"
                    issuer: "https://token.actions.githubusercontent.com"
                    rekor:
                      url: https://rekor.sigstore.dev
```

### 5.2 OPA/Gatekeeper ポリシー

```yaml
# ConstraintTemplate: 信頼済みレジストリのみ許可
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8strustedregistries
spec:
  crd:
    spec:
      names:
        kind: K8sTrustedRegistries
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
        package trustedregistries
        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not trusted(container.image)
          msg := sprintf("Image '%v' is not from a trusted registry", [container.image])
        }
        trusted(image) {
          registry := input.parameters.registries[_]
          startswith(image, registry)
        }

---
# Constraint: 適用
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sTrustedRegistries
metadata:
  name: require-trusted-registry
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
  parameters:
    registries:
      - "ghcr.io/myorg/"
      - "gcr.io/myproject/"
```

---

## 6. サプライチェーンセキュリティツール一覧

```
+------------------------------------------------------------------+
|              サプライチェーンセキュリティツール                       |
+------------------------------------------------------------------+
|                                                                  |
|  カテゴリ          | ツール         | 用途                       |
|  ------------------|---------------|---------------------------|
|  イメージ署名       | cosign        | 署名・検証                 |
|                    | Notation      | OCI 署名 (MS/AWS推進)      |
|  SBOM 生成         | Trivy         | スキャン + SBOM            |
|                    | Syft          | SBOM 生成専用              |
|  脆弱性スキャン     | Trivy         | 包括的スキャン             |
|                    | Grype         | SBOM ベーススキャン        |
|  来歴 (Provenance)  | SLSA Generator| ビルド来歴証明             |
|                    | in-toto       | ビルドステップ証明         |
|  透明性ログ         | Rekor         | 署名の公開ログ             |
|                    | Fulcio        | 短命証明書の発行           |
|  ポリシー適用       | Kyverno       | K8s ポリシー (署名検証)    |
|                    | OPA/Gatekeeper| K8s ポリシー (汎用)        |
|  シークレット検出    | gitleaks      | Git 履歴のシークレット検出  |
|                    | TruffleHog    | シークレット検出 (高精度)   |
|                                                                  |
+------------------------------------------------------------------+
```

---

## アンチパターン

### アンチパターン 1: タグでイメージを参照する

```yaml
# NG: タグは上書き可能。改ざんリスクがある
containers:
  - name: app
    image: ghcr.io/myorg/myapp:latest
    # "latest" タグが別のイメージに差し替えられる可能性

# OK: ダイジェスト (SHA256) で参照
containers:
  - name: app
    image: ghcr.io/myorg/myapp@sha256:a1b2c3d4e5f6...
    # ダイジェストはイメージの内容に対する一意なハッシュ
    # 改ざんされていれば検証で失敗する
```

**問題点**: タグ (`latest`, `v1.0.0` 等) はレジストリ上で任意のイメージに再割り当てできる。攻撃者がレジストリに侵入した場合、タグを悪意のあるイメージに向けることが可能。ダイジェスト参照はイメージの内容 (SHA256 ハッシュ) を直接指定するため、改ざんが不可能。

### アンチパターン 2: SBOM を生成するだけで活用しない

```bash
# NG: SBOM を生成して放置
trivy image --format spdx-json -o sbom.json myapp:latest
# → ファイルが生成されるだけで誰も見ない

# OK: SBOM を継続的に活用するパイプライン
# 1. SBOM 生成 + イメージに添付
trivy image --format spdx-json -o sbom.json myapp:latest
cosign attach sbom --sbom sbom.json ghcr.io/myorg/myapp@sha256:xxx

# 2. 定期的な脆弱性再スキャン (新しい CVE の検出)
trivy sbom sbom.json  # 新しい脆弱性DBで再スキャン

# 3. ライセンスコンプライアンスチェック
trivy sbom --scanners license sbom.json

# 4. 依存関係の棚卸し (EOL パッケージの検出)
```

**問題点**: SBOM は「生成すること」が目的ではなく、「継続的に脆弱性を追跡すること」が目的。新しい CVE が公開された際に、影響を受けるイメージを特定するためのデータベースとして活用する。

---

## FAQ

### Q1: cosign の Keyless 署名はどのような仕組みですか？

**A**: Keyless 署名では長期的な秘密鍵を管理する必要がない。仕組みは (1) CI/CD プラットフォーム (GitHub Actions 等) が OIDC トークンを発行、(2) Fulcio (Sigstore の CA) がトークンを検証し、短命の署名用証明書 (10分間有効) を発行、(3) その証明書で署名を行い、(4) 署名と証明書が Rekor (透明性ログ) に記録される。検証時は Rekor ログと OIDC の issuer/subject を確認する。鍵管理の負担がなく、CI/CD 環境に最適。

### Q2: SBOM のフォーマットは SPDX と CycloneDX のどちらを選ぶべきですか？

**A**: 米国政府関連や法規制対応 (EO 14028) が必要な場合は SPDX。セキュリティ分析が主目的なら CycloneDX。実務的には多くのツール (Trivy, Grype, Syft) が両方をサポートしているため、消費するツールチェーンとの互換性で選択する。CycloneDX は VEX (Vulnerability Exploitability eXchange) との統合が優れており、「この脆弱性は我々の環境では影響なし」という判断を SBOM に組み込める。

### Q3: サプライチェーンセキュリティの導入を段階的に進めるにはどうすべきですか？

**A**: 推奨する段階的導入: (1) まず Trivy でイメージスキャンを CI/CD に追加 (1日で導入可能)、(2) SBOM 生成を CI/CD に追加し、成果物として保存 (半日)、(3) cosign Keyless 署名を導入 (1日)、(4) ダイジェスト参照に移行 (1-2日)、(5) Admission Controller (Kyverno) で署名検証をステージング環境に導入 (1-2日)、(6) 本番環境への展開。全てを一度に導入しようとせず、各段階で効果を確認しながら進める。

---

## まとめ

| 項目 | 要点 |
|------|------|
| cosign | Sigstore プロジェクトのイメージ署名ツール。Keyless 推奨 |
| Keyless 署名 | OIDC + Fulcio + Rekor で鍵管理不要の署名を実現 |
| SBOM | SPDX / CycloneDX 形式でソフトウェア構成を記録 |
| Trivy + Syft | SBOM 生成と脆弱性スキャンの主要ツール |
| Provenance | SLSA フレームワークでビルド来歴を記録・検証 |
| ダイジェスト参照 | タグではなく SHA256 ダイジェストでイメージを参照 |
| Admission Controller | Kyverno / OPA で署名済みイメージのみデプロイ許可 |
| 段階的導入 | スキャン → SBOM → 署名 → ポリシー適用の順で導入 |

## 次に読むべきガイド

- [コンテナセキュリティ](./00-container-security.md) -- イメージスキャン、最小権限、Dockerfile のセキュリティ
- [Kubernetes 応用](../05-orchestration/02-kubernetes-advanced.md) -- Helm / Ingress / ConfigMap の本番運用
- Docker Compose セキュリティ -- 開発環境でのセキュリティ意識

## 参考文献

1. **Sigstore 公式** -- https://www.sigstore.dev/ -- cosign, Fulcio, Rekor を含む Sigstore プロジェクトの全体像
2. **SLSA フレームワーク** -- https://slsa.dev/ -- Supply chain Levels for Software Artifacts の仕様と実装ガイド
3. **SPDX Specification** -- https://spdx.dev/ -- SBOM の SPDX フォーマット仕様
4. **Kyverno 公式** -- https://kyverno.io/ -- Kubernetes ポリシーエンジンによるイメージ署名検証
5. **NIST SSDF (SP 800-218)** -- https://csrc.nist.gov/Projects/ssdf -- セキュアソフトウェア開発フレームワーク

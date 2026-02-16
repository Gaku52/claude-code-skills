# サプライチェーンセキュリティ (Supply Chain Security)

> コンテナイメージの署名 (cosign)、SBOM (Software Bill of Materials) の生成・検証を通じて、ビルドからデプロイまでのソフトウェアサプライチェーン全体の信頼性を確保する手法を学ぶ。

## この章で学ぶこと

1. **イメージ署名と検証 (cosign / Sigstore)** -- コンテナイメージにデジタル署名を付与し、改ざんされていないことを検証する仕組みを構築する
2. **SBOM の生成と活用** -- ソフトウェアの構成要素を一覧化し、脆弱性の追跡と規制対応を実現する
3. **CI/CD パイプラインでのサプライチェーン保護** -- ビルドの来歴 (Provenance) 記録と、ポリシーベースのデプロイ制御を実装する
4. **VEX (Vulnerability Exploitability eXchange)** -- 脆弱性の影響度判断を SBOM と連携して管理する
5. **依存関係の安全管理** -- パッケージの改ざん防止、typosquatting 対策、lockfile の重要性

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

### 1.1 主要なサプライチェーン攻撃事例

近年のサプライチェーン攻撃は深刻化している。以下は代表的な事例と教訓である。

| 事例 | 年 | 攻撃手法 | 教訓 |
|------|-----|---------|------|
| SolarWinds | 2020 | ビルドシステムへの侵入 | ビルド来歴の記録が重要 |
| Codecov | 2021 | CI スクリプトの改ざん | 実行スクリプトの完全性検証 |
| Log4Shell | 2021 | 依存ライブラリの脆弱性 | SBOM で影響範囲を即座に特定 |
| ua-parser-js | 2021 | npm パッケージの乗っ取り | lockfile + 署名検証 |
| colors/faker | 2022 | メンテナによる意図的破壊 | 依存関係のバージョン固定 |
| xz-utils | 2024 | メンテナへのソーシャルエンジニアリング | コードレビュー + 再現可能ビルド |

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
|       |     GitHub Actions -> Fulcio -> Rekor (透明性ログ)        |
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

### 2.2 Sigstore プロジェクトの構成

```
+------------------------------------------------------------------+
|              Sigstore エコシステム                                  |
+------------------------------------------------------------------+
|                                                                  |
|  cosign: イメージ署名・検証ツール                                  |
|    -> CLI でイメージに署名 / 検証 / SBOM 添付                     |
|                                                                  |
|  Fulcio: 証明書発行局 (CA)                                       |
|    -> OIDC トークンを検証し、短命 (10分) の署名証明書を発行        |
|    -> 長期的な秘密鍵の管理が不要になる                             |
|                                                                  |
|  Rekor: 透明性ログ (Transparency Log)                            |
|    -> 全ての署名を不変のログに記録                                 |
|    -> 署名の存在証明と監査が可能                                   |
|    -> Certificate Transparency と同じ概念                         |
|                                                                  |
|  policy-controller: Kubernetes Admission Controller              |
|    -> デプロイ時に署名を自動検証                                   |
|    -> 未署名イメージのデプロイを拒否                               |
|                                                                  |
|  Gitsign: Git コミットの署名                                      |
|    -> Keyless で Git コミットに署名                                |
|    -> GPG 鍵の管理が不要                                          |
|                                                                  |
+------------------------------------------------------------------+
```

### 2.3 cosign のインストールと鍵生成

```bash
# インストール
# macOS
brew install cosign

# Linux
curl -L https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64 \
  -o /usr/local/bin/cosign && chmod +x /usr/local/bin/cosign

# バージョン確認
cosign version

# 鍵ペア生成 (Key-pair 方式)
cosign generate-key-pair
# -> cosign.key (秘密鍵、パスワード保護)
# -> cosign.pub (公開鍵、配布用)

# KMS を使う場合
cosign generate-key-pair --kms awskms:///alias/cosign-key
cosign generate-key-pair --kms gcpkms://projects/myproject/locations/global/keyRings/myring/cryptoKeys/mykey
cosign generate-key-pair --kms azurekms://myvault.vault.azure.net/keys/mykey

# Kubernetes Secret に鍵を保存
cosign generate-key-pair k8s://production/cosign-key
```

### 2.4 イメージ署名と検証

```bash
# イメージのビルドとプッシュ
docker build -t ghcr.io/myorg/myapp:1.0.0 .
docker push ghcr.io/myorg/myapp:1.0.0

# ダイジェストを取得 (タグではなくダイジェストで署名)
DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/myorg/myapp:1.0.0)
echo $DIGEST
# -> ghcr.io/myorg/myapp@sha256:abc123...

# Key-pair 方式で署名
cosign sign --key cosign.key $DIGEST

# Keyless 方式で署名 (推奨、CI/CD 向け)
cosign sign --yes $DIGEST

# カスタムアノテーション付きで署名
cosign sign --yes \
  -a "commit=$(git rev-parse HEAD)" \
  -a "build-url=https://github.com/myorg/myapp/actions/runs/12345" \
  -a "build-date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  $DIGEST

# 検証 (Key-pair)
cosign verify --key cosign.pub ghcr.io/myorg/myapp:1.0.0

# 検証 (Keyless)
cosign verify \
  --certificate-identity="https://github.com/myorg/myapp/.github/workflows/build.yml@refs/tags/v1.0.0" \
  --certificate-oidc-issuer="https://token.actions.githubusercontent.com" \
  ghcr.io/myorg/myapp:1.0.0

# 検証 (正規表現でマッチ)
cosign verify \
  --certificate-identity-regexp="https://github.com/myorg/.*" \
  --certificate-oidc-issuer="https://token.actions.githubusercontent.com" \
  ghcr.io/myorg/myapp:1.0.0

# 署名の詳細表示
cosign tree ghcr.io/myorg/myapp:1.0.0
```

### 2.5 GitHub Actions での Keyless 署名

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
            -a "commit=${{ github.sha }}" \
            -a "ref=${{ github.ref }}" \
            -a "workflow=${{ github.workflow }}" \
            -a "run-id=${{ github.run_id }}" \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}

      # SBOM を生成して添付
      - name: Generate and attach SBOM
        run: |
          # Trivy で SBOM 生成
          trivy image --format spdx-json \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }} \
            > sbom.spdx.json

          # SBOM をイメージに添付
          cosign attach sbom \
            --sbom sbom.spdx.json \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}

          # SBOM 自体にも署名
          cosign sign --yes \
            --attachment sbom \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}

      # Attestation の添付 (カスタムメタデータ)
      - name: Attest build
        run: |
          cosign attest --yes \
            --type custom \
            --predicate <(cat <<EOF
          {
            "buildType": "github-actions",
            "builder": {
              "id": "https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }}"
            },
            "metadata": {
              "buildStartedOn": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
              "completeness": {
                "parameters": true,
                "environment": true,
                "materials": true
              }
            }
          }
          EOF
          ) \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}

      # 検証ステップ (確認)
      - name: Verify signature
        run: |
          cosign verify \
            --certificate-identity-regexp="https://github.com/${{ github.repository }}/.*" \
            --certificate-oidc-issuer="https://token.actions.githubusercontent.com" \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}
```

### 2.6 GitLab CI での署名

```yaml
# .gitlab-ci.yml
build-and-sign:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  id_tokens:
    SIGSTORE_ID_TOKEN:
      aud: sigstore
  variables:
    IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_TAG
  script:
    - docker build -t $IMAGE .
    - docker push $IMAGE
    - DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' $IMAGE)
    # cosign のインストール
    - apk add --no-cache cosign
    # Keyless 署名 (GitLab OIDC トークンを使用)
    - cosign sign --yes $DIGEST
  rules:
    - if: $CI_COMMIT_TAG
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
|  |     +-- openssl 3.1.4 <- CVE-2024-XXXX (修正済み)             |
|  |     +-- ca-certificates 20240226                              |
|  |     +-- libcrypto3 3.1.4                                      |
|  |     +-- libssl3 3.1.4                                         |
|  |     +-- ...                                                   |
|  |                                                               |
|  +-- アプリ依存関係                                               |
|  |     +-- express 4.18.2                                        |
|  |     +-- pg 8.11.3                                             |
|  |     +-- @prisma/client 5.7.1                                  |
|  |     +-- winston 3.11.0                                        |
|  |     +-- helmet 7.1.0                                          |
|  |     +-- ...                                                   |
|  |                                                               |
|  +-- ライセンス情報                                               |
|  |     +-- express: MIT                                          |
|  |     +-- pg: MIT                                               |
|  |     +-- @prisma/client: Apache-2.0                            |
|  |     +-- ...                                                   |
|  |                                                               |
|  +-- メタデータ                                                   |
|       +-- ビルド日時: 2025-01-15T10:30:00Z                       |
|       +-- ビルドツール: docker buildx 0.12.0                     |
|       +-- ソース: github.com/myorg/myapp@abc1234                 |
|       +-- SBOM 生成ツール: trivy 0.50.0                          |
|                                                                  |
+------------------------------------------------------------------+
```

SBOM は「ソフトウェアの成分表」である。食品の原材料表示と同じように、ソフトウェアに含まれる全てのコンポーネント (OS パッケージ、ライブラリ、フレームワーク) とそのバージョン、ライセンス情報を一覧化する。新しい脆弱性が公開された際に、影響を受けるソフトウェアを即座に特定できる。

### 3.2 SBOM 生成ツール

```bash
# Trivy で SBOM 生成 (SPDX 形式)
trivy image --format spdx-json --output sbom-spdx.json myapp:latest

# Trivy で SBOM 生成 (CycloneDX 形式)
trivy image --format cyclonedx --output sbom-cdx.json myapp:latest

# Syft で SBOM 生成 (Anchore 製)
syft myapp:latest -o spdx-json > sbom-syft-spdx.json
syft myapp:latest -o cyclonedx-json > sbom-syft-cdx.json

# Syft で特定のスコープを指定
syft myapp:latest -o spdx-json --scope all-layers > sbom-all-layers.json

# Docker Scout (Docker Desktop 統合)
docker scout sbom myapp:latest
docker scout sbom --format spdx myapp:latest

# BuildKit で SBOM を自動生成
docker buildx build --sbom=true -t myapp:latest .

# ローカルディレクトリから SBOM 生成
syft dir:. -o cyclonedx-json > sbom-source.json
trivy fs --format spdx-json --output sbom-fs.json .
```

### 3.3 SBOM 形式の比較

| 項目 | SPDX | CycloneDX | SWID |
|------|------|-----------|------|
| 標準化団体 | Linux Foundation | OWASP | ISO/IEC |
| フォーマット | JSON, RDF, Tag-Value | JSON, XML, Protobuf | XML |
| ライセンス情報 | 非常に詳細 | 基本的 | 基本的 |
| 脆弱性情報 | 外部連携 | VEX 統合 | なし |
| 依存関係グラフ | サポート | サポート | 限定的 |
| サービス情報 | 限定的 | 詳細 (API, エンドポイント) | なし |
| 採用事例 | 米国政府 (EO 14028) | セキュリティツール | レガシー |
| ツール対応 | 広い | 広い | 限定的 |
| 推奨用途 | ライセンスコンプライアンス | セキュリティ分析 | 資産管理 |

### 3.4 SBOM からの脆弱性スキャン

```bash
# SBOM を入力として脆弱性スキャン
trivy sbom sbom-spdx.json

# 重大度でフィルタ
trivy sbom --severity CRITICAL,HIGH sbom-spdx.json

# JSON 出力
trivy sbom --format json --output vuln-results.json sbom-spdx.json

# grype で SBOM スキャン
grype sbom:sbom-cdx.json

# grype で重大度フィルタ
grype sbom:sbom-cdx.json --fail-on critical

# ライセンスコンプライアンスチェック
trivy sbom --scanners license sbom-spdx.json

# 特定のライセンスを検出
trivy sbom --scanners license --severity HIGH sbom-spdx.json
```

### 3.5 SBOM の継続的管理

```yaml
# .github/workflows/sbom-management.yml
name: SBOM Management

on:
  push:
    branches: [main]
  schedule:
    # 毎日再スキャン (新しい CVE の検出)
    - cron: '0 6 * * *'

jobs:
  generate-sbom:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Generate SBOM
        run: |
          # SPDX 形式
          trivy image --format spdx-json \
            --output sbom-spdx.json \
            ghcr.io/${{ github.repository }}:latest

          # CycloneDX 形式
          trivy image --format cyclonedx \
            --output sbom-cdx.json \
            ghcr.io/${{ github.repository }}:latest

      - name: Scan SBOM for vulnerabilities
        run: |
          trivy sbom --severity CRITICAL,HIGH \
            --exit-code 1 \
            sbom-spdx.json

      - name: Check licenses
        run: |
          trivy sbom --scanners license \
            --severity HIGH \
            sbom-spdx.json

      - name: Upload SBOM as artifact
        uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: |
            sbom-spdx.json
            sbom-cdx.json
          retention-days: 90

      - name: Attach SBOM to release
        if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')
        run: |
          gh release upload ${{ github.ref_name }} \
            sbom-spdx.json \
            sbom-cdx.json
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## 4. VEX (Vulnerability Exploitability eXchange)

### 4.1 VEX とは

VEX は「この脆弱性は我々の環境では影響しない」という判断を機械可読な形式で表現する仕組みである。SBOM に対する脆弱性スキャンで大量の CVE が検出されても、全てが実際に悪用可能とは限らない。VEX により、セキュリティチームの判断を記録・共有できる。

```
+------------------------------------------------------------------+
|              VEX のステータス                                      |
+------------------------------------------------------------------+
|                                                                  |
|  Not Affected (影響なし):                                        |
|    -> 脆弱なコードパスが到達不能                                   |
|    -> 脆弱な機能を使用していない                                   |
|    例: OpenSSL の DTLS 脆弱性だが、DTLS を使っていない              |
|                                                                  |
|  Affected (影響あり):                                             |
|    -> 脆弱性の影響を受ける                                        |
|    -> 修正対応が必要                                              |
|                                                                  |
|  Fixed (修正済み):                                                |
|    -> 修正バージョンに更新済み                                     |
|                                                                  |
|  Under Investigation (調査中):                                    |
|    -> 影響の有無を調査中                                          |
|                                                                  |
+------------------------------------------------------------------+
```

### 4.2 VEX ドキュメントの作成

```json
{
  "@context": "https://openvex.dev/ns/v0.2.0",
  "@id": "https://myorg.example.com/vex/2025-01-15",
  "author": "security-team@example.com",
  "timestamp": "2025-01-15T10:00:00Z",
  "statements": [
    {
      "vulnerability": {
        "@id": "https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
        "name": "CVE-2024-12345",
        "description": "OpenSSL DTLS vulnerability"
      },
      "products": [
        {
          "@id": "pkg:oci/myapp@sha256:abc123",
          "subcomponents": [
            {"@id": "pkg:apk/alpine/openssl@3.1.4"}
          ]
        }
      ],
      "status": "not_affected",
      "justification": "vulnerable_code_not_in_execute_path",
      "impact_statement": "This application does not use DTLS. The vulnerable code path is never executed."
    },
    {
      "vulnerability": {
        "@id": "https://nvd.nist.gov/vuln/detail/CVE-2024-67890",
        "name": "CVE-2024-67890"
      },
      "products": [
        {"@id": "pkg:oci/myapp@sha256:abc123"}
      ],
      "status": "affected",
      "action_statement": "Update to express@4.19.0 which contains the fix.",
      "action_statement_timestamp": "2025-01-20T00:00:00Z"
    }
  ]
}
```

### 4.3 VEX と Trivy の連携

```bash
# VEX ファイルを使ったスキャン
trivy image --vex vex.json myapp:latest

# VEX で "not_affected" とされた CVE はスキャン結果から除外される
# -> ノイズが減り、本当に対応すべき脆弱性に集中できる
```

---

## 5. ビルド来歴 (Provenance)

### 5.1 SLSA (Supply chain Levels for Software Artifacts)

```
+------------------------------------------------------------------+
|              SLSA レベル                                          |
+------------------------------------------------------------------+
|                                                                  |
|  Level 0: なし                                                   |
|    -> 何もしない                                                  |
|                                                                  |
|  Level 1: 来歴の存在                                             |
|    -> ビルドプロセスの記録が存在する                                |
|    -> 手動ビルドでも可                                             |
|    -> 最小要件: ビルドスクリプトがバージョン管理されている           |
|                                                                  |
|  Level 2: ホスティングされたビルド                                 |
|    -> CI/CD サービスでビルド                                       |
|    -> 来歴が自動生成される                                         |
|    -> 要件: ビルドが CI/CD プラットフォーム上で実行される            |
|                                                                  |
|  Level 3: ハード化されたビルド                                     |
|    -> ビルド環境が改ざん耐性を持つ                                  |
|    -> 来歴が暗号署名される                                         |
|    -> ビルドジョブ間のパラメータ注入が防止される                    |
|    -> GitHub Actions + SLSA Generator が対応                      |
|                                                                  |
|  Level 4 (将来): 完全な来歴                                       |
|    -> 二者レビュー                                                |
|    -> 密閉型ビルド (ネットワーク分離)                              |
|    -> 再現可能ビルド                                              |
|                                                                  |
+------------------------------------------------------------------+
```

### 5.2 GitHub Actions での Provenance 生成

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

### 5.3 SLSA Generator の使用

```yaml
# .github/workflows/slsa-generator.yml
name: SLSA Level 3 Build

on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      image: ${{ steps.build.outputs.image }}
      digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: build
        run: |
          IMAGE=ghcr.io/${{ github.repository }}:${{ github.ref_name }}
          docker build -t $IMAGE .
          docker push $IMAGE
          DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' $IMAGE | cut -d@ -f2)
          echo "image=$IMAGE" >> $GITHUB_OUTPUT
          echo "digest=$DIGEST" >> $GITHUB_OUTPUT

  # SLSA Generator で Level 3 の Provenance を生成
  provenance:
    needs: build
    permissions:
      actions: read
      id-token: write
      packages: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v2.0.0
    with:
      image: ${{ needs.build.outputs.image }}
      digest: ${{ needs.build.outputs.digest }}
      registry-username: ${{ github.actor }}
    secrets:
      registry-password: ${{ secrets.GITHUB_TOKEN }}
```

### 5.4 in-toto によるビルドステップ証明

```bash
# in-toto: 各ビルドステップの証明を生成
# Step 1: ソースコードのチェックアウト
in-toto-run --step-name checkout \
  --products src/ \
  --signing-key developer.key \
  -- git clone https://github.com/myorg/myapp.git src/

# Step 2: テスト
in-toto-run --step-name test \
  --materials src/ \
  --signing-key ci.key \
  -- cd src && npm test

# Step 3: ビルド
in-toto-run --step-name build \
  --materials src/ \
  --products dist/ \
  --signing-key ci.key \
  -- cd src && npm run build

# レイアウト (ポリシー) の検証
in-toto-verify --layout root.layout \
  --layout-key owner.pub \
  --verification-keys developer.pub ci.pub
```

---

## 6. 依存関係の安全管理

### 6.1 lockfile による依存関係の固定

```bash
# npm: package-lock.json
npm ci  # lockfile に厳密に従ってインストール (npm install は使わない)

# pnpm: pnpm-lock.yaml
pnpm install --frozen-lockfile

# yarn: yarn.lock
yarn install --frozen-lockfile

# pip: requirements.txt (ハッシュ付き)
pip install --require-hashes -r requirements.txt

# Go: go.sum
go mod verify
```

```text
# requirements.txt (ハッシュ付き - 改ざん検出)
flask==3.0.0 \
    --hash=sha256:21128f47e4e3b9d29ce5c59c0ab98341a9f8e8da8e1da9ffa6b8651d2d8f3a5c
requests==2.31.0 \
    --hash=sha256:58cd2187c01e70e6e26505bca751777aa9f2ee0b7f4300988b709f44e013003eb
```

### 6.2 依存関係の監査

```bash
# npm audit (脆弱性チェック)
npm audit
npm audit --audit-level=high
npm audit fix  # 自動修正

# pip-audit (Python)
pip install pip-audit
pip-audit -r requirements.txt

# cargo audit (Rust)
cargo install cargo-audit
cargo audit

# Trivy でファイルシステムスキャン
trivy fs --scanners vuln .

# Renovate / Dependabot で自動更新 PR
```

### 6.3 Renovate による自動更新

```json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": [
    "config:recommended",
    "security:openssf-scorecard"
  ],
  "labels": ["dependencies"],
  "vulnerabilityAlerts": {
    "enabled": true,
    "labels": ["security"]
  },
  "packageRules": [
    {
      "matchUpdateTypes": ["minor", "patch"],
      "automerge": true,
      "automergeType": "pr",
      "requiredStatusChecks": ["ci"]
    },
    {
      "matchUpdateTypes": ["major"],
      "automerge": false,
      "reviewers": ["team:platform"]
    },
    {
      "matchPackagePatterns": ["^@types/"],
      "automerge": true,
      "schedule": ["before 9am on monday"]
    }
  ],
  "docker": {
    "pinDigests": true
  },
  "helm-values": {
    "fileMatch": ["values.*\\.yaml$"]
  }
}
```

### 6.4 Dependabot の設定

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    reviewers:
      - "myorg/platform-team"
    labels:
      - "dependencies"
    open-pull-requests-limit: 10
    groups:
      production-dependencies:
        patterns:
          - "*"
        exclude-patterns:
          - "@types/*"
          - "eslint*"
          - "*jest*"
      dev-dependencies:
        patterns:
          - "@types/*"
          - "eslint*"
          - "*jest*"

  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

---

## 7. デプロイ時のポリシー適用

### 7.1 Kubernetes Admission Controller

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

### 7.2 Sigstore policy-controller

```bash
# policy-controller のインストール
helm repo add sigstore https://sigstore.github.io/helm-charts
helm install policy-controller sigstore/policy-controller \
  -n cosign-system --create-namespace
```

```yaml
# ClusterImagePolicy: 署名検証ポリシー
apiVersion: policy.sigstore.dev/v1alpha1
kind: ClusterImagePolicy
metadata:
  name: require-signed-images
spec:
  images:
    - glob: "ghcr.io/myorg/**"
  authorities:
    - keyless:
        identities:
          - issuer: "https://token.actions.githubusercontent.com"
            subject: "https://github.com/myorg/*/.github/workflows/*"
        ctlog:
          url: "https://rekor.sigstore.dev"

---
# Namespace に適用
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    policy.sigstore.dev/include: "true"
```

### 7.3 OPA/Gatekeeper ポリシー

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
        violation[{"msg": msg}] {
          container := input.review.object.spec.initContainers[_]
          not trusted(container.image)
          msg := sprintf("Init container image '%v' is not from a trusted registry", [container.image])
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
    namespaces: ["production", "staging"]
  parameters:
    registries:
      - "ghcr.io/myorg/"
      - "gcr.io/myproject/"
```

### 7.4 ダイジェスト固定ポリシー

```yaml
# Kyverno ポリシー: タグではなくダイジェストでの参照を強制
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-image-digest
spec:
  validationFailureAction: Enforce
  rules:
    - name: check-digest
      match:
        any:
          - resources:
              kinds:
                - Pod
              namespaces:
                - production
      validate:
        message: "Images must be referenced by digest (@sha256:...), not by tag."
        pattern:
          spec:
            containers:
              - image: "*@sha256:*"
```

---

## 8. サプライチェーンセキュリティツール一覧

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
|                    | docker sbom   | Docker Desktop 統合        |
|  脆弱性スキャン     | Trivy         | 包括的スキャン             |
|                    | Grype         | SBOM ベーススキャン        |
|                    | Snyk          | 開発者向けスキャン         |
|  VEX              | OpenVEX       | 脆弱性影響度判断           |
|                    | CycloneDX VEX | CycloneDX 統合            |
|  来歴 (Provenance)  | SLSA Generator| ビルド来歴証明             |
|                    | in-toto       | ビルドステップ証明         |
|  透明性ログ         | Rekor         | 署名の公開ログ             |
|                    | Fulcio        | 短命証明書の発行           |
|  ポリシー適用       | Kyverno       | K8s ポリシー (署名検証)    |
|                    | OPA/Gatekeeper| K8s ポリシー (汎用)        |
|                    | policy-controller | Sigstore 統合ポリシー  |
|  シークレット検出    | gitleaks      | Git 履歴のシークレット検出  |
|                    | TruffleHog    | シークレット検出 (高精度)   |
|  依存関係管理       | Renovate      | 自動更新 PR               |
|                    | Dependabot    | GitHub 統合自動更新        |
|  Dockerfile Lint   | hadolint      | Dockerfile 品質チェック    |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 9. レジストリのセキュリティ設定

### 9.1 イミュータブルタグの設定

```bash
# ECR: イミュータブルタグの有効化
aws ecr put-image-tag-mutability \
  --repository-name myapp \
  --image-tag-mutability IMMUTABLE

# GHCR: タグの上書き防止 (組織設定)
# GitHub Organization Settings -> Packages -> Default repository permissions
# "Prevent forked repositories from creating packages" を有効化
```

### 9.2 脆弱性スキャンの有効化

```bash
# ECR: スキャン設定
aws ecr put-image-scanning-configuration \
  --repository-name myapp \
  --image-scanning-configuration scanOnPush=true

# GCR: Container Analysis の有効化
gcloud services enable containeranalysis.googleapis.com
gcloud services enable containerscanning.googleapis.com
```

### 9.3 レジストリのアクセス制御

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowPushFromCI",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:role/github-actions-role"
      },
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:CompleteLayerUpload",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart"
      ]
    },
    {
      "Sid": "AllowPullFromEKS",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:role/eks-node-role"
      },
      "Action": [
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ]
    },
    {
      "Sid": "DenyDeleteImages",
      "Effect": "Deny",
      "Principal": "*",
      "Action": [
        "ecr:BatchDeleteImage",
        "ecr:DeleteRepository"
      ],
      "Condition": {
        "StringNotLike": {
          "aws:PrincipalArn": "arn:aws:iam::123456789012:role/admin-role"
        }
      }
    }
  ]
}
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

# NG: セマンティックバージョンタグでも同じ問題
containers:
  - name: app
    image: ghcr.io/myorg/myapp:1.0.0
    # タグは再割り当て可能

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
# -> ファイルが生成されるだけで誰も見ない

# OK: SBOM を継続的に活用するパイプライン
# 1. SBOM 生成 + イメージに添付
trivy image --format spdx-json -o sbom.json myapp:latest
cosign attach sbom --sbom sbom.json ghcr.io/myorg/myapp@sha256:xxx

# 2. 定期的な脆弱性再スキャン (新しい CVE の検出)
trivy sbom sbom.json  # 新しい脆弱性DBで再スキャン

# 3. ライセンスコンプライアンスチェック
trivy sbom --scanners license sbom.json

# 4. 依存関係の棚卸し (EOL パッケージの検出)

# 5. VEX で影響度を判断・記録
```

**問題点**: SBOM は「生成すること」が目的ではなく、「継続的に脆弱性を追跡すること」が目的。新しい CVE が公開された際に、影響を受けるイメージを特定するためのデータベースとして活用する。

### アンチパターン 3: lockfile を Git にコミットしない

```bash
# NG: lockfile が .gitignore に含まれている
# .gitignore
package-lock.json
pnpm-lock.yaml
yarn.lock

# OK: lockfile は必ず Git にコミット
# lockfile により、全ての開発者と CI/CD で同じ依存関係が再現される
# lockfile なしでは、ビルドごとに異なるバージョンがインストールされるリスクがある
```

**問題点**: lockfile がないと、`npm install` のたびに依存関係のバージョンが変わる可能性がある。攻撃者がパッケージの新バージョンに悪意のあるコードを挿入した場合、lockfile がなければ自動的にそのバージョンがインストールされる。`npm ci` (lockfile に厳密に従うインストール) を CI/CD で使用し、lockfile を必ずバージョン管理する。

### アンチパターン 4: 署名検証を本番でのみ実施する

```yaml
# NG: 本番環境でのみ署名検証
# -> ステージングで動作確認 -> 本番で署名検証に失敗 -> デプロイできない

# OK: ステージングから署名検証を適用
# ステージング: Audit モード (検証失敗をログに記録するが拒否しない)
# 本番: Enforce モード (検証失敗時にデプロイを拒否)
```

**問題点**: 署名検証を本番でのみ有効にすると、ステージングでは動作確認できたのに本番でデプロイが拒否されるという事態が発生する。ステージング環境でも署名検証を実施し、問題を早期に発見する。

---

## FAQ

### Q1: cosign の Keyless 署名はどのような仕組みですか？

**A**: Keyless 署名では長期的な秘密鍵を管理する必要がない。仕組みは (1) CI/CD プラットフォーム (GitHub Actions 等) が OIDC トークンを発行、(2) Fulcio (Sigstore の CA) がトークンを検証し、短命の署名用証明書 (10分間有効) を発行、(3) その証明書で署名を行い、(4) 署名と証明書が Rekor (透明性ログ) に記録される。検証時は Rekor ログと OIDC の issuer/subject を確認する。鍵管理の負担がなく、CI/CD 環境に最適。

### Q2: SBOM のフォーマットは SPDX と CycloneDX のどちらを選ぶべきですか？

**A**: 米国政府関連や法規制対応 (EO 14028) が必要な場合は SPDX。セキュリティ分析が主目的なら CycloneDX。実務的には多くのツール (Trivy, Grype, Syft) が両方をサポートしているため、消費するツールチェーンとの互換性で選択する。CycloneDX は VEX (Vulnerability Exploitability eXchange) との統合が優れており、「この脆弱性は我々の環境では影響なし」という判断を SBOM に組み込める。

### Q3: サプライチェーンセキュリティの導入を段階的に進めるにはどうすべきですか？

**A**: 推奨する段階的導入: (1) まず Trivy でイメージスキャンを CI/CD に追加 (1日で導入可能)、(2) SBOM 生成を CI/CD に追加し、成果物として保存 (半日)、(3) cosign Keyless 署名を導入 (1日)、(4) ダイジェスト参照に移行 (1-2日)、(5) Admission Controller (Kyverno) で署名検証をステージング環境に導入 (1-2日)、(6) 本番環境への展開。全てを一度に導入しようとせず、各段階で効果を確認しながら進める。

### Q4: ダイジェスト参照にすると、バージョンが分かりにくくなります。どう管理すべきですか？

**A**: (1) Kyverno の `mutate` ルールを使い、タグをダイジェストに自動変換する。(2) Flux CD や Argo CD の image automation 機能を使い、新しいイメージが push されたら自動的にダイジェスト参照を更新する。(3) コメントでタグ情報を残す (`# v1.2.0` のようなアノテーション)。(4) Renovate / Dependabot にダイジェスト固定と自動更新を任せる。実運用ではツールによる自動化が不可欠。

### Q5: Notation (ORAS/OCI 署名) と cosign の違いは何ですか？

**A**: cosign は Sigstore プロジェクトの一部で、Keyless 署名 (OIDC ベース) が最大の特徴。Notation は Microsoft と AWS が推進する OCI 署名仕様で、既存の PKI インフラ (X.509 証明書) との親和性が高い。cosign は CI/CD 環境での自動署名に優れ、Notation はエンタープライズの既存証明書基盤との統合に優れる。現時点では cosign のエコシステムが充実しているが、Notation も成熟が進んでいる。

### Q6: OpenSSF Scorecard とは何ですか？

**A**: OpenSSF Scorecard は OSS プロジェクトのセキュリティプラクティスを自動評価するツール。CI/CD のセキュリティ、ブランチ保護、コードレビュー、依存関係の管理状況などを 0-10 のスコアで評価する。`scorecard --repo=github.com/myorg/myapp` で実行でき、GitHub Actions にも統合できる。自分のプロジェクトのセキュリティ成熟度を客観的に把握し、改善点を特定するのに有効。

---

## まとめ

| 項目 | 要点 |
|------|------|
| cosign | Sigstore プロジェクトのイメージ署名ツール。Keyless 推奨 |
| Keyless 署名 | OIDC + Fulcio + Rekor で鍵管理不要の署名を実現 |
| SBOM | SPDX / CycloneDX 形式でソフトウェア構成を記録 |
| VEX | 脆弱性の影響度を機械可読な形式で判断・共有 |
| Trivy + Syft | SBOM 生成と脆弱性スキャンの主要ツール |
| Provenance | SLSA フレームワークでビルド来歴を記録・検証 |
| ダイジェスト参照 | タグではなく SHA256 ダイジェストでイメージを参照 |
| Admission Controller | Kyverno / OPA / policy-controller で署名済みイメージのみデプロイ許可 |
| lockfile | 依存関係を固定し、再現可能なビルドを実現 |
| 依存関係管理 | Renovate / Dependabot で自動更新 + 脆弱性アラート |
| 段階的導入 | スキャン -> SBOM -> 署名 -> ポリシー適用の順で導入 |

## 次に読むべきガイド

- [コンテナセキュリティ](./00-container-security.md) -- イメージスキャン、最小権限、Dockerfile のセキュリティ
- [Kubernetes 応用](../05-orchestration/02-kubernetes-advanced.md) -- Helm / Ingress / ConfigMap の本番運用
- Docker Compose セキュリティ -- 開発環境でのセキュリティ意識

## 参考文献

1. **Sigstore 公式** -- https://www.sigstore.dev/ -- cosign, Fulcio, Rekor を含む Sigstore プロジェクトの全体像
2. **SLSA フレームワーク** -- https://slsa.dev/ -- Supply chain Levels for Software Artifacts の仕様と実装ガイド
3. **SPDX Specification** -- https://spdx.dev/ -- SBOM の SPDX フォーマット仕様
4. **CycloneDX** -- https://cyclonedx.org/ -- OWASP 主導の SBOM フォーマットと VEX 統合
5. **Kyverno 公式** -- https://kyverno.io/ -- Kubernetes ポリシーエンジンによるイメージ署名検証
6. **NIST SSDF (SP 800-218)** -- https://csrc.nist.gov/Projects/ssdf -- セキュアソフトウェア開発フレームワーク
7. **OpenVEX** -- https://openvex.dev/ -- VEX の仕様と実装ガイド
8. **OpenSSF Scorecard** -- https://scorecard.dev/ -- OSS プロジェクトのセキュリティ評価ツール

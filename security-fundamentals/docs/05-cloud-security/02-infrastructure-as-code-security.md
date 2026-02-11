# IaC セキュリティ

> tfsec、Checkov によるインフラコードの自動セキュリティチェック、ポリシー as コードによるガバナンス適用まで、Infrastructure as Code のセキュリティを体系的に学ぶ

## この章で学ぶこと

1. **IaC のセキュリティリスク** — Terraform/CloudFormation コードに潜む設定ミスとその影響
2. **静的解析ツール** — tfsec、Checkov、KICS によるセキュリティポリシーの自動検証
3. **ポリシー as コード** — OPA/Rego、Sentinel によるカスタムポリシーの実装

---

## 1. IaC セキュリティの重要性

### IaC で起きるセキュリティ問題

```
+----------------------------------------------------------+
|         IaC の典型的なセキュリティ問題                       |
|----------------------------------------------------------|
|                                                          |
|  [ネットワーク]                                            |
|  +-- Security Group で 0.0.0.0/0:22 を許可               |
|  +-- NACL のデフォルト全許可                               |
|  +-- VPC ピアリングの過剰な許可                            |
|                                                          |
|  [データ保護]                                              |
|  +-- S3 バケットのパブリックアクセス                        |
|  +-- RDS/EBS の暗号化未設定                               |
|  +-- ログの暗号化未設定                                    |
|                                                          |
|  [認証・認可]                                              |
|  +-- IAM ポリシーの * (全許可)                             |
|  +-- ハードコードされた認証情報                             |
|  +-- MFA 未設定のリソース                                  |
|                                                          |
|  [ログ・監視]                                              |
|  +-- CloudTrail 無効                                     |
|  +-- VPC Flow Logs 未設定                                 |
|  +-- アクセスログ未有効化                                   |
+----------------------------------------------------------+
```

### IaC のセキュリティチェックのタイミング

```
開発者 PC         CI/CD              デプロイ前            ランタイム
    |                |                    |                    |
  [pre-commit]    [ビルド]            [Plan/Apply]         [ドリフト検知]
    |                |                    |                    |
  tfsec           Checkov             Sentinel/OPA         AWS Config
  (IDE連携)       KICS                (ポリシーゲート)       Prowler
                  tfsec                                     (定期スキャン)
```

---

## 2. tfsec (Terraform セキュリティスキャナ)

### tfsec の使い方

```bash
# インストール
brew install tfsec

# スキャン実行
tfsec .

# 特定の重大度以上のみ
tfsec --minimum-severity HIGH .

# JSON 出力 (CI/CD 向け)
tfsec --format json --out results.json .

# SARIF 出力 (GitHub Security タブ連携)
tfsec --format sarif --out results.sarif .
```

### tfsec の検出例と修正

```hcl
# NG: tfsec が検出する問題
resource "aws_s3_bucket" "data" {
  bucket = "my-data-bucket"
  # aws-s3-enable-bucket-encryption: 暗号化未設定
  # aws-s3-enable-bucket-logging: アクセスログ未設定
  # aws-s3-enable-versioning: バージョニング未設定
}

resource "aws_security_group_rule" "ssh" {
  type        = "ingress"
  from_port   = 22
  to_port     = 22
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]  # aws-vpc-no-public-ingress-sgr
}

# OK: 修正後
resource "aws_s3_bucket" "data" {
  bucket = "my-data-bucket"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.data.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_logging" "data" {
  bucket        = aws_s3_bucket.data.id
  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "s3-access-logs/"
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

### tfsec のインライン抑制

```hcl
# 特定のルールを正当な理由で抑制
resource "aws_security_group_rule" "https" {
  type        = "ingress"
  from_port   = 443
  to_port     = 443
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]  #tfsec:ignore:aws-vpc-no-public-ingress-sgr -- Public HTTPS endpoint
}
```

---

## 3. Checkov (マルチフレームワーク対応)

### Checkov の特徴

| 項目 | tfsec | Checkov | KICS |
|------|-------|---------|------|
| 対応 IaC | Terraform | Terraform, CFn, K8s, ARM, Docker | 多数 |
| ルール数 | ~1000 | ~2000 | ~2000 |
| カスタムポリシー | YAML/Rego | Python/YAML | Rego |
| グラフベース解析 | 部分的 | あり (依存関係解析) | なし |
| SCA 機能 | なし | あり | なし |
| CI/CD 統合 | GitHub Action | GitHub Action, pre-commit | GitHub Action |

### Checkov の使い方

```bash
# インストール
pip install checkov

# Terraform スキャン
checkov -d . --framework terraform

# Kubernetes マニフェストスキャン
checkov -d ./k8s/ --framework kubernetes

# Dockerfile スキャン
checkov --file Dockerfile --framework dockerfile

# 特定のチェックのみ実行
checkov -d . --check CKV_AWS_18,CKV_AWS_19,CKV_AWS_21

# 特定のチェックをスキップ
checkov -d . --skip-check CKV_AWS_999

# 出力形式
checkov -d . -o json > checkov-results.json
checkov -d . -o sarif > checkov-results.sarif
```

### Checkov カスタムポリシー (Python)

```python
# custom_checks/s3_naming_convention.py
from checkov.terraform.checks.resource.base_resource_check import BaseResourceCheck
from checkov.common.models.enums import CheckResult, CheckCategories

class S3NamingConvention(BaseResourceCheck):
    """S3 バケット名が命名規則に従っているか"""

    def __init__(self):
        name = "S3 bucket follows naming convention"
        id = "CKV_CUSTOM_1"
        supported_resources = ["aws_s3_bucket"]
        categories = [CheckCategories.CONVENTION]
        super().__init__(name=name, id=id, categories=categories,
                        supported_resources=supported_resources)

    def scan_resource_conf(self, conf):
        bucket_name = conf.get("bucket", [""])[0]
        # 命名規則: {env}-{service}-{purpose}
        if any(prefix in bucket_name for prefix in ["prod-", "stg-", "dev-"]):
            return CheckResult.PASSED
        return CheckResult.FAILED

check = S3NamingConvention()
```

### CI/CD 統合

```yaml
# GitHub Actions での Checkov + tfsec 統合
name: IaC Security
on:
  pull_request:
    paths: ['terraform/**', 'k8s/**']

jobs:
  iac-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: tfsec
        uses: aquasecurity/tfsec-action@v1.0.0
        with:
          working_directory: terraform/
          soft_fail: false

      - name: Checkov
        uses: bridgecrewio/checkov-action@master
        with:
          directory: terraform/
          framework: terraform
          output_format: sarif
          output_file_path: checkov.sarif
          soft_fail: false

      - name: Upload SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: checkov.sarif
```

---

## 4. ポリシー as コード (OPA / Sentinel)

### OPA (Open Policy Agent) + Rego

```rego
# policy/terraform/s3.rego
package terraform.s3

# S3 バケットの暗号化を必須化
deny[msg] {
    resource := input.resource.aws_s3_bucket[name]
    not has_encryption(name)
    msg := sprintf("S3 bucket '%s' must have server-side encryption enabled", [name])
}

has_encryption(bucket_name) {
    input.resource.aws_s3_bucket_server_side_encryption_configuration[_].bucket == bucket_name
}

# パブリックアクセスブロックを必須化
deny[msg] {
    resource := input.resource.aws_s3_bucket[name]
    not has_public_access_block(name)
    msg := sprintf("S3 bucket '%s' must have public access block", [name])
}

has_public_access_block(bucket_name) {
    block := input.resource.aws_s3_bucket_public_access_block[_]
    block.bucket == bucket_name
    block.block_public_acls == true
    block.block_public_policy == true
}
```

### Conftest による OPA ポリシーテスト

```bash
# Terraform plan を JSON に変換
terraform plan -out=tfplan
terraform show -json tfplan > tfplan.json

# OPA ポリシーでテスト
conftest test tfplan.json --policy policy/

# 出力例:
# FAIL - tfplan.json - terraform.s3 - S3 bucket 'data' must have encryption
# FAIL - tfplan.json - terraform.s3 - S3 bucket 'data' must have public access block
# 2 tests, 0 passed, 2 warnings, 2 failures
```

### ポリシーテスト体系

```
policy/
  ├── terraform/
  │   ├── s3.rego           # S3 ポリシー
  │   ├── iam.rego          # IAM ポリシー
  │   ├── network.rego      # ネットワークポリシー
  │   └── encryption.rego   # 暗号化ポリシー
  ├── kubernetes/
  │   ├── pod_security.rego
  │   └── network_policy.rego
  └── tests/
      ├── s3_test.rego      # ポリシーのユニットテスト
      └── iam_test.rego
```

---

## 5. ドリフト検知

### ドリフトとは

```
IaC コード (あるべき状態)     実際のインフラ (現在の状態)
+-----------------------+    +-----------------------+
| SG: port 443 のみ許可 |    | SG: port 443 + 22     |
|                       | != |  (手動で SSH 追加)     |
| 暗号化: 有効           |    | 暗号化: 有効           |
+-----------------------+    +-----------------------+
                                  ↑
                              ドリフト (乖離)
```

```bash
# Terraform でドリフト検知
terraform plan -detailed-exitcode
# Exit code 2 = ドリフトあり

# AWS Config でドリフト検知 (CloudFormation)
aws cloudformation detect-stack-drift --stack-name my-stack

# driftctl (専用ツール)
driftctl scan --from tfstate://terraform.tfstate
```

---

## 6. アンチパターン

### アンチパターン 1: Terraform state ファイルの不安全な管理

```hcl
# NG: ローカルに state を保存 (暗号化なし、共有不可)
terraform {
  backend "local" {}
}

# OK: リモート state + 暗号化 + ロック
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "ap-northeast-1"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:ap-northeast-1:123456:key/xxx"
    dynamodb_table = "terraform-state-lock"
  }
}
```

**影響**: state ファイルにはシークレット情報が含まれる可能性がある。漏洩するとインフラ全体の設定が攻撃者に露出する。

### アンチパターン 2: IaC スキャンの CI/CD 非統合

```
NG: 開発者がローカルでのみスキャンを実行
  → 忘れたり、結果を無視したりする

OK: CI/CD でスキャンを強制し、失敗時はマージをブロック
  → PR のマージ条件に tfsec/Checkov のパスを含める
  → Branch Protection Rule で必須ステータスチェックに設定
```

---

## 7. FAQ

### Q1. tfsec と Checkov のどちらを使うべきか?

Terraform のみを使っている場合は tfsec が高速でシンプルである。Terraform に加えて Kubernetes、Dockerfile、CloudFormation なども管理している場合は Checkov のマルチフレームワーク対応が有利である。両方を併用しても問題はない。

### Q2. OPA ポリシーの管理はどうすべきか?

ポリシーは専用の Git リポジトリで管理し、CI/CD で自動テストする。ポリシーの変更にもレビュープロセスを適用する。OPA のテストフレームワーク (`opa test`) でポリシーのユニットテストを書き、意図しない許可・拒否を防ぐ。

### Q3. 既存インフラを IaC 化する際のセキュリティ考慮は?

`terraform import` で既存リソースを IaC に取り込んだ後、即座に tfsec/Checkov でスキャンする。多数のセキュリティ問題が見つかる場合は優先度をつけて段階的に修正する。ドリフト検知を有効にして IaC 外の手動変更を検出する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| IaC のリスク | 設定ミスがコードに残り、大規模に展開される |
| tfsec | Terraform 特化の高速スキャナ |
| Checkov | マルチフレームワーク対応の包括的スキャナ |
| OPA/Rego | カスタムポリシーの定義と自動適用 |
| CI/CD 統合 | PR のマージ条件にスキャンパスを必須化 |
| ドリフト検知 | IaC と実インフラの乖離を継続的に検出 |
| State 管理 | リモート + 暗号化 + ロックで安全に管理 |

---

## 次に読むべきガイド

- [AWSセキュリティ](./01-aws-security.md) — IaC で設定する AWS セキュリティサービス
- [クラウドセキュリティ基礎](./00-cloud-security-basics.md) — IAM と暗号化の基礎
- [コンテナセキュリティ](../04-application-security/02-container-security.md) — Kubernetes マニフェストのセキュリティ

---

## 参考文献

1. **Checkov Documentation** — https://www.checkov.io/
2. **tfsec Documentation** — https://aquasecurity.github.io/tfsec/
3. **Open Policy Agent (OPA) Documentation** — https://www.openpolicyagent.org/docs/latest/
4. **HashiCorp Sentinel Documentation** — https://developer.hashicorp.com/sentinel/docs
